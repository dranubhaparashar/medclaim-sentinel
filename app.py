from __future__ import annotations

import inspect
import json
from pathlib import Path

import pandas as pd
import plotly.express as px
import streamlit as st
from PIL import Image

from medclaim import db
from medclaim.pipeline import load_sample_claim, process_claim, process_document, run_local_ai_for_claim, save_uploaded_file
from medclaim.services.chatbot import answer_question
from medclaim.services.local_ai import get_ollama_status, load_config, save_config

st.set_page_config(page_title="MedClaim Sentinel", page_icon="🩺", layout="wide")
db.init_db()

STATUS_ORDER = [
    "Received", "Documents Processing", "OCR Review Required", "Ready for Assessment",
    "Possible Duplicate", "Medical Review Required", "Additional Information Required",
    "Supervisor Review", "Approved", "Partially Approved", "Rejected", "Closed", "Reopened",
]

LEGACY_CLINICAL_CODES = {"TEST_INDICATION_CHECK", "PRESCRIPTION_PRESENT", "DATE_COVERAGE"}
DIMENSION_SUMMARY_CODES = {
    "DIAGNOSIS_EVIDENCE_SUMMARY",
    "DIAGNOSIS_PRESCRIPTION_SUMMARY",
    "PRESCRIPTION_BILL_SUMMARY",
    "DIAGNOSIS_INVESTIGATION_SUMMARY",
}
SUMMARY_CODES = {"CLINICAL_MATCH_SUMMARY", "LOCAL_AI_CLINICAL_SUMMARY"} | DIMENSION_SUMMARY_CODES
STATUS_META = {
    "MATCHED": ("✅", "Matched"),
    "PARTIALLY_MATCHED": ("🟡", "Partially matched"),
    "CONDITIONAL_MATCH": ("🟡", "Conditional match"),
    "MISMATCH": ("🔴", "Possible mismatch — medical review"),
    "INSUFFICIENT_EVIDENCE": ("📄", "Insufficient evidence"),
    "MEDICAL_OFFICER_REVIEW": ("🟠", "Medical-officer review"),
    "INFORMATION": ("ℹ️", "Information"),
    "consistent": ("✅", "Matched"),
    "information": ("ℹ️", "Information"),
    "review": ("🟠", "Review"),
}


def process_claim_compat(claim_id: int, *, run_local_clinical: bool = False):
    """Run only the v7 deterministic pipeline unless local AI is explicitly requested.

    Older v6 pipeline builds always called Ollama from ``process_claim``. Falling back
    to those builds made Claim 360 appear frozen during ordinary page navigation.
    This patch ships the compatible v7 pipeline and refuses silent fallback.
    """
    parameters = inspect.signature(process_claim).parameters
    if "run_local_clinical" not in parameters:
        raise RuntimeError(
            "medclaim/pipeline.py is outdated. Copy the complete v7.3 patch, "
            "including medclaim/pipeline.py, into the project folder."
        )
    return process_claim(claim_id, run_local_clinical=run_local_clinical)


def _check_status(check):
    evidence = check.get("evidence") or {}
    if isinstance(evidence, dict):
        return str(evidence.get("clinical_status") or check.get("result") or "INFORMATION")
    return str(check.get("result") or "INFORMATION")


def _status_text(status):
    icon, label = STATUS_META.get(str(status), ("🟠", str(status).replace("_", " ").title()))
    return f"{icon} {label}"


def _confidence_text(value):
    try:
        number = float(value)
    except (TypeError, ValueError):
        return "Not scored"
    if number <= 0:
        return "Not scored"
    return f"{number:.0%}"


def _dimension_card(check):
    evidence = check.get("evidence") or {}
    status = _check_status(check)
    st.markdown(f"### {_status_text(status)}")
    st.caption(check.get("explanation") or "")
    confidence = evidence.get("confidence") if isinstance(evidence, dict) else None
    st.write(f"**Evidence confidence:** {_confidence_text(confidence)}")
    required = evidence.get("required_evidence") if isinstance(evidence, dict) else None
    if required:
        st.write(f"**Needed:** {required}")


def _aggregate_clinical_rows(checks):
    grouped = {}
    for check in checks:
        code = str(check.get("check_code") or "")
        if code in SUMMARY_CODES or code.startswith("LOCAL_AI_"):
            continue
        evidence = check.get("evidence") or {}
        if not isinstance(evidence, dict):
            evidence = {"raw_evidence": evidence}
        status = _check_status(check)
        if status == "INFORMATION":
            continue
        dimension = evidence.get("dimension") or code
        item = evidence.get("item") or "—"
        key = (dimension, item, status)
        confidence = evidence.get("confidence")
        raw = evidence.get("raw_evidence")
        dates = []
        if isinstance(raw, dict):
            test = raw.get("test")
            if isinstance(test, dict) and test.get("date"):
                dates.append(str(test.get("date")))
        row = grouped.setdefault(key, {
            "Dimension": dimension,
            "Item": item,
            "Status": _status_text(status),
            "Confidence values": [],
            "Occurrences": 0,
            "Dates": [],
            "Finding": check.get("explanation") or "",
            "Required evidence": evidence.get("required_evidence") or "—",
            "Reference": evidence.get("source_title") or "—",
            "Open reference": evidence.get("source_url") or "",
        })
        row["Occurrences"] += 1
        if confidence not in (None, ""):
            try:
                row["Confidence values"].append(float(confidence))
            except (TypeError, ValueError):
                pass
        row["Dates"].extend(dates)
    rows = []
    for row in grouped.values():
        values = row.pop("Confidence values")
        row["Evidence confidence"] = _confidence_text(sum(values) / len(values)) if values else "Not scored"
        row["Dates"] = ", ".join(dict.fromkeys(row["Dates"])) or "—"
        rows.append(row)
    return rows


def _aggregate_status(statuses):
    """Return the most conservative status for a group of evidence rows."""
    statuses = [str(x or "") for x in statuses if x]
    if not statuses:
        return "INSUFFICIENT_EVIDENCE"
    priority = [
        "MISMATCH",
        "MEDICAL_OFFICER_REVIEW",
        "INSUFFICIENT_EVIDENCE",
        "PARTIALLY_MATCHED",
        "CONDITIONAL_MATCH",
        "MATCHED",
        "INFORMATION",
    ]
    for status in priority:
        if status in statuses:
            return status
    return statuses[0]


def _dimension_fallbacks(checks):
    """Reconstruct summary cards from saved evidence when an older DB lacks summary rows.

    v7 introduced four explicit summary checks. Claims processed by an intermediate
    build can contain the new evidence-level checks but not those summary records.
    The UI must not call such claims "Not assessed" when evidence was actually assessed.
    """
    fallbacks = {}
    overall = next((c for c in checks if c.get("check_code") == "CLINICAL_MATCH_SUMMARY"), None)
    overall_evidence = (overall or {}).get("evidence") or {}
    saved_statuses = overall_evidence.get("dimension_statuses") if isinstance(overall_evidence, dict) else {}
    saved_statuses = saved_statuses if isinstance(saved_statuses, dict) else {}

    config = [
        (
            "DIAGNOSIS_EVIDENCE_SUMMARY",
            "1. Diagnosis / treatment context",
            "diagnosis",
            {"Diagnosis evidence"},
        ),
        (
            "DIAGNOSIS_PRESCRIPTION_SUMMARY",
            "2. Diagnosis ↔ prescribed medicine",
            "diagnosis_prescription",
            {"Diagnosis ↔ prescribed medicine"},
        ),
        (
            "PRESCRIPTION_BILL_SUMMARY",
            "3. Prescribed medicine ↔ billed medicine",
            "prescription_bill",
            {"Prescription ↔ pharmacy bill"},
        ),
        (
            "DIAGNOSIS_INVESTIGATION_SUMMARY",
            "4. Diagnosis ↔ investigation",
            "diagnosis_investigation",
            {"Diagnosis ↔ investigation"},
        ),
    ]

    for code, title, overall_key, dimensions in config:
        rows = []
        for check in checks:
            evidence = check.get("evidence") or {}
            if not isinstance(evidence, dict):
                continue
            if evidence.get("dimension") in dimensions:
                rows.append(check)

        status = saved_statuses.get(overall_key)
        if not status:
            if code == "DIAGNOSIS_EVIDENCE_SUMMARY":
                if any(c.get("check_code") == "DIAGNOSIS_CONTEXT" for c in checks):
                    status = "MATCHED"
                elif any(c.get("check_code") == "DIAGNOSIS_MISSING" for c in checks):
                    status = "INSUFFICIENT_EVIDENCE"
            if not status:
                status = _aggregate_status([_check_status(row) for row in rows])

        confidences = []
        required = []
        for row in rows:
            evidence = row.get("evidence") or {}
            try:
                if evidence.get("confidence") not in (None, ""):
                    confidences.append(float(evidence.get("confidence")))
            except (TypeError, ValueError):
                pass
            needed = evidence.get("required_evidence")
            if needed and needed not in required:
                required.append(str(needed))

        if rows:
            explanation = f"Reconstructed from {len(rows)} saved evidence finding(s). Refresh clinical rules to persist the four summary records."
        elif status == "MATCHED" and code == "DIAGNOSIS_EVIDENCE_SUMMARY":
            explanation = "A diagnosis/treatment context was recognized from the saved claim evidence."
        else:
            explanation = "The saved claim does not contain enough evidence to assess this relationship."

        fallbacks[code] = {
            "check_code": code,
            "severity": "medium",
            "result": "consistent" if status == "MATCHED" else "review",
            "explanation": explanation,
            "evidence": {
                "clinical_status": status,
                "dimension": title,
                "confidence": (sum(confidences) / len(confidences)) if confidences else None,
                "required_evidence": "; ".join(required) if required else None,
                "reconstructed": True,
            },
        }
    return fallbacks


def _render_overall_status(summary, prefix="Rule-engine assessment"):
    if not summary:
        st.info("Clinical matching has not been generated yet. Run the clinical rules.")
        return
    status = _check_status(summary)
    message = f"**{prefix}: {_status_text(status)}** — {summary.get('explanation') or ''}"
    if status == "MATCHED":
        st.success(message)
    elif status in {"MISMATCH", "MEDICAL_OFFICER_REVIEW"}:
        st.error(message)
    else:
        st.warning(message)


def money(v):
    return f"₹{float(v or 0):,.0f}"


def badge(text: str):
    st.markdown(f"**{text}**")


def run_local_ai_with_progress(claim_id: int, *, force: bool = False):
    """Run resumable local-AI processing and show the current page in Streamlit."""
    status_box = st.status("Preparing local AI…", expanded=True)
    progress_bar = st.progress(0.0)
    current_line = st.empty()

    def on_progress(done: int, total: int, filename: str, state: str):
        fraction = 1.0 if total <= 0 else max(0.0, min(float(done) / float(total), 1.0))
        progress_bar.progress(fraction)
        if state == "processing":
            current_line.write(f"Reading {filename} ({min(done + 1, total)}/{total})…")
        elif state == "completed":
            status_box.write(f"✓ {filename}")
        elif state == "skipped":
            status_box.write(f"↷ {filename} already processed; skipped")
        elif state == "failed":
            status_box.write(f"⚠ {filename} failed")
        elif state == "clinical":
            current_line.write("Running text-only clinical matching from extracted evidence…")

    try:
        result = run_local_ai_for_claim(
            claim_id,
            force=force,
            progress_callback=on_progress,
        )
    except Exception as exc:
        status_box.update(label="Local AI stopped with an error", state="error", expanded=True)
        st.error(str(exc))
        return None

    progress_bar.progress(1.0)
    current_line.empty()
    summary = f"Processed {result.get('processed', 0)}, skipped {result.get('skipped', 0)}"
    if result.get("failures"):
        status_box.update(label=summary + "; some pages failed", state="error", expanded=True)
        st.warning("; ".join(f"{x['filename']}: {x['error']}" for x in result["failures"]))
    else:
        status_box.update(label=summary + "; clinical checks refreshed", state="complete", expanded=False)
    return result


def claim_selector(key: str = "claim_select"):
    claims = db.list_claims()
    if not claims:
        st.info("No claims are available. Load the included sample or create a new claim.")
        return None
    mapping = {f"{c['claim_no']} — {c['patient_name']}": c["id"] for c in claims}
    label = st.selectbox("Select claim", list(mapping), key=key)
    return mapping[label]


def _claim_is_incomplete(claim, document_count: int) -> bool:
    return (
        document_count == 0
        or str(claim.get("patient_name") or "").strip() in {"", "Pending OCR Review"}
        or float(claim.get("claimed_amount") or 0) <= 0
        or str(claim.get("status") or "") in {"Received", "Documents Processing", "OCR Review Required"}
    )


def _delete_claim_panel(claim_id: int, *, key_prefix: str) -> None:
    claim = db.get_claim(claim_id)
    if not claim:
        st.info("This claim no longer exists.")
        return
    documents = db.list_documents(claim_id)
    incomplete = _claim_is_incomplete(claim, len(documents))

    if incomplete:
        st.warning(
            "This entry appears incomplete or half-loaded. Deleting it removes the claim, "
            "its uploaded files, OCR output, clinical checks, duplicate links, corrections and audit events."
        )
    else:
        st.warning(
            "Permanent deletion removes the complete claim and its uploaded evidence. "
            "Use this only when the entry was created by mistake or must be removed under an approved process."
        )

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Claim", str(claim.get("claim_no") or "—"))
    c2.metric("Patient", str(claim.get("patient_name") or "—"))
    c3.metric("Documents", len(documents))
    c4.metric("Status", str(claim.get("status") or "—"))

    default_reason = "Incomplete/half-loaded entry" if incomplete else "Entry created by mistake"
    reason = st.text_input(
        "Reason for deletion",
        value=default_reason,
        key=f"{key_prefix}_delete_reason",
    )
    expected = str(claim.get("claim_no") or "")
    confirmation = st.text_input(
        f"Type `{expected}` to confirm",
        key=f"{key_prefix}_delete_confirmation",
        placeholder=expected,
    )
    acknowledged = st.checkbox(
        "I understand that this permanently removes the claim and its uploaded files.",
        key=f"{key_prefix}_delete_ack",
    )
    ready = acknowledged and confirmation.strip() == expected and bool(reason.strip())

    if st.button(
        "Permanently delete claim",
        type="primary",
        disabled=not ready,
        key=f"{key_prefix}_delete_button",
    ):
        with st.spinner(f"Deleting {expected}..."):
            result = db.delete_claim(
                claim_id,
                deleted_by="claim_reviewer",
                reason=reason.strip(),
                delete_files=True,
            )
            refreshed = 0
            refresh_errors = []
            for affected_claim_id in result.get("affected_claim_ids", []):
                try:
                    process_claim_compat(int(affected_claim_id), run_local_clinical=False)
                    refreshed += 1
                except Exception as exc:
                    refresh_errors.append(str(exc))

        for state_key in (
            "claim_360_selector",
            "dashboard_delete_selector",
            f"{key_prefix}_delete_confirmation",
            f"{key_prefix}_delete_ack",
            f"{key_prefix}_delete_reason",
        ):
            st.session_state.pop(state_key, None)
        message = (
            f"Deleted {result.get('claim_no')}. Removed "
            f"{result.get('record_counts', {}).get('documents', 0)} document record(s) "
            f"and {result.get('removed_files', 0)} stored file(s)."
        )
        if refreshed:
            message += f" Recalculated {refreshed} linked claim(s)."
        if refresh_errors or result.get("file_errors"):
            message += " Some cleanup warnings were recorded; check the terminal."
        st.session_state["claim_delete_flash"] = message
        st.rerun()


def render_claim_management(claims=None) -> None:
    claims = claims or db.list_claims()
    st.subheader("Manage claim entries")
    st.caption("Remove accidental, test, incomplete or half-loaded claims from the active system.")
    if not claims:
        st.info("No claims are available to delete.")
        return

    metadata = {}
    for claim in claims:
        document_count = len(db.list_documents(int(claim["id"])))
        incomplete = _claim_is_incomplete(claim, document_count)
        marker = "Incomplete" if incomplete else "Complete"
        label = f"{claim['claim_no']} — {claim['patient_name']} — {marker} — {document_count} document(s)"
        metadata[label] = int(claim["id"])

    show_incomplete_only = st.checkbox(
        "Show only incomplete/half-loaded entries",
        value=True,
        key="dashboard_delete_incomplete_only",
    )
    options = list(metadata)
    if show_incomplete_only:
        filtered = []
        for label, claim_id in metadata.items():
            claim = next(item for item in claims if int(item["id"]) == claim_id)
            if _claim_is_incomplete(claim, len(db.list_documents(claim_id))):
                filtered.append(label)
        options = filtered

    if not options:
        st.info("No incomplete entries were found. Turn off the filter to manage all claims.")
        return

    selected_label = st.selectbox(
        "Select entry",
        options,
        key="dashboard_delete_selector",
    )
    _delete_claim_panel(metadata[selected_label], key_prefix="dashboard")


def render_dashboard():
    st.title("MedClaim Sentinel")
    st.caption("Local multimodal AI for claim OCR, duplicate detection, diagnosis–medicine matching, finance and review chat")
    flash = st.session_state.pop("claim_delete_flash", None)
    if flash:
        st.success(flash)
    claims = db.list_claims()
    if not claims:
        st.info("Start from **New Claim** and load the included sample.")
        return
    df = pd.DataFrame(claims)
    c1, c2, c3, c4, c5 = st.columns(5)
    c1.metric("Claims", len(df))
    c2.metric("Claimed", money(df["claimed_amount"].sum()))
    c3.metric("Supported", money(df["supported_amount"].sum()))
    c4.metric("Possible duplicates", int(df["status"].eq("Possible Duplicate").sum()))
    c5.metric("Medical review", int(df["status"].eq("Medical Review Required").sum()))

    left, right = st.columns([1.2, 1])
    with left:
        st.subheader("Pipeline queue")
        view = df[["claim_no", "patient_name", "claimed_amount", "supported_amount", "risk_score", "risk_band", "status"]].copy()
        st.dataframe(view, use_container_width=True, hide_index=True)
    with right:
        st.subheader("Claims by status")
        counts = df.groupby("status", as_index=False).size()
        fig = px.bar(counts, x="status", y="size", labels={"size": "Claims", "status": "Status"})
        st.plotly_chart(fig, use_container_width=True)

    st.subheader("Risk distribution")
    risk_counts = df.groupby("risk_band", as_index=False).size()
    fig = px.pie(risk_counts, names="risk_band", values="size")
    st.plotly_chart(fig, use_container_width=True)

    st.divider()
    with st.expander("Delete or clean up claim entries", expanded=False):
        render_claim_management(claims)


def render_new_claim():
    st.title("New Claim")
    st.write("Upload the complete claim bundle. The local vision-language model reads the actual images, extracts diagnosis/medicine/test/bill evidence and runs clinical consistency checks automatically.")
    ai_status = get_ollama_status()
    if ai_status["available"] and ai_status["requested_model_installed"]:
        pass
    elif ai_status["available"]:
        st.warning(f"Ollama is running, but `{ai_status['vision_model']}` is not installed. Run `ollama pull {ai_status['vision_model']}` before processing for automatic image extraction.")
    else:
        st.warning("Local AI is not connected. The claim can still be uploaded with basic OCR, but automatic medicine/diagnosis matching needs Ollama. Run `install_local_ai.ps1` once.")
    if st.button("Load included sample claim", type="primary"):
        with st.spinner("Loading and processing sample documents..."):
            claim_id = load_sample_claim()
        st.success(f"Sample claim loaded. Database claim ID: {claim_id}")

    st.divider()
    generated_claim_no = f"CLM-{len(db.list_claims()) + 1:05d}"
    with st.form("new_claim_form"):
        st.markdown(f"**System-generated claim number:** `{generated_claim_no}`")
        uploads = st.file_uploader(
            "Claim documents",
            type=["png", "jpg", "jpeg", "tif", "tiff"],
            accept_multiple_files=True,
            help="Upload all pages for one application. Patient name and claim metadata are read from the documents.",
        )
        with st.expander("Optional manual details / OCR override"):
            c1, c2 = st.columns(2)
            with c1:
                application_no = st.text_input("Application number")
                patient_name = st.text_input("Patient name", help="Leave blank to detect it from the documents.")
                policy_no = st.text_input("Policy/member number")
            with c2:
                hospital = st.text_input("Hospital / provider")
                diagnosis = st.text_area("Diagnosis / treatment context")
                claimed_amount = st.number_input("Claimed amount (INR)", min_value=0.0, step=100.0, help="Leave 0 to calculate it from extracted bill totals.")
        submitted = st.form_submit_button("Upload, OCR and process claim", type="primary")

    if submitted:
        if not uploads:
            st.error("Please upload at least one claim document.")
            return
        claim_no = generated_claim_no
        while db.claim_exists(claim_no):
            next_number = int(claim_no.split("-")[-1]) + 1
            claim_no = f"CLM-{next_number:05d}"

        claim_id = db.create_claim({
            "claim_no": claim_no,
            "application_no": application_no,
            "patient_name": patient_name.strip() or "Pending OCR Review",
            "policy_no": policy_no,
            "hospital": hospital,
            "diagnosis": diagnosis,
            "claimed_amount": claimed_amount,
            "status": "Documents Processing",
        })
        progress = st.progress(0, text="Saving and reading documents...")
        for i, upload in enumerate(uploads, start=1):
            path = save_uploaded_file(claim_id, upload.name, upload.getvalue())
            process_document(claim_id, path, upload.name)
            progress.progress(i / len(uploads), text=f"Processed {i} of {len(uploads)} document(s)")
        result = process_claim_compat(claim_id)
        detected = db.get_claim(claim_id) or {}
        progress.empty()
        st.success(f"Claim {claim_no} processed: {result['status']}; risk {result['risk']['score']}/100.")
        if detected.get("patient_name") == "Pending OCR Review":
            st.warning("The handwriting reader could not verify the name. The claim was still created and does not need to be uploaded again.")
        else:
            st.write(f"**Detected patient:** {detected.get('patient_name')}")
        st.write(f"**Detected provider:** {detected.get('hospital') or 'Needs review'}")
        st.write(f"**Detected/derived claimed amount:** {money(detected.get('claimed_amount'))}")
        processed_docs = db.list_documents(claim_id)
        render_identity_review(detected, processed_docs, key_prefix=f"new_claim_{claim_id}")
        st.info("Open **Claim 360** for document OCR, duplicate evidence, clinical checks, finance and audit history.")



def render_identity_review(claim, docs, key_prefix: str = "identity"):
    st.subheader("Patient identity review")
    st.caption("Handwritten names and signatures are evidence, not a safe basis for automatic identity unless corroborated by a member/policy record or a high-confidence local vision result.")

    candidates = []
    crops = []
    local_vision = None
    for doc in docs:
        ext = doc.get("extracted", {}) or {}
        for candidate in ext.get("identity_candidates", []) or []:
            if isinstance(candidate, dict) and candidate.get("value"):
                candidates.append({
                    "Candidate": candidate.get("value"),
                    "Confidence": float(candidate.get("confidence", 0) or 0),
                    "Source": candidate.get("source", doc.get("filename")),
                    "Document": doc.get("filename"),
                })
        for crop in ext.get("identity_crops", []) or []:
            if isinstance(crop, dict) and crop.get("path"):
                crops.append(dict(crop, document=doc.get("filename")))
        if ext.get("local_vision"):
            local_vision = ext.get("local_vision")

    current = claim.get("patient_name") or "Pending OCR Review"
    if current == "Pending OCR Review":
        st.warning("The name was not read with sufficient confidence. The claim is preserved; confirm the name here without uploading again.")
    else:
        st.success(f"Current patient/claimant name: {current}")

    if candidates:
        ranked = pd.DataFrame(candidates).sort_values("Confidence", ascending=False).drop_duplicates(subset=["Candidate", "Document"])
        st.markdown("**OCR name candidates — do not accept automatically without checking the image**")
        st.dataframe(ranked, use_container_width=True, hide_index=True)

    visible_crops = [c for c in crops if Path(c["path"]).exists()][:6]
    if visible_crops:
        st.markdown("**Name/signature evidence regions**")
        cols = st.columns(2)
        for i, crop in enumerate(visible_crops):
            with cols[i % 2]:
                st.image(crop["path"], use_container_width=True)
                st.caption(f"{crop.get('document')} — {crop.get('label')}")

    with st.form(f"{key_prefix}_confirm_name_form"):
        default_name = "" if current == "Pending OCR Review" else current
        confirmed_name = st.text_input("Confirmed patient/claimant name", value=default_name, key=f"{key_prefix}_confirmed_name")
        reviewer_note = st.text_input("Review note", value="Confirmed from uploaded claim documents", key=f"{key_prefix}_review_note")
        save_name = st.form_submit_button("Confirm and save name", type="primary")
    if save_name:
        cleaned = " ".join(confirmed_name.split()).strip()
        if len(cleaned) < 3:
            st.error("Enter the readable patient/claimant name before saving.")
        else:
            previous = claim.get("patient_name")
            db.update_claim(claim["id"], patient_name=cleaned)
            db.audit(claim["id"], "claim_reviewer", "PATIENT_NAME_CONFIRMED", {"from": previous, "to": cleaned, "note": reviewer_note})
            st.success(f"Saved patient/claimant name: {cleaned}")
            st.rerun()

    st.info("Patient identity is read automatically by the configured local vision model. Low-confidence handwriting remains marked for review rather than being silently accepted.")

def render_summary(claim, docs, duplicates, checks):
    st.subheader("Claim summary")
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Claimed", money(claim["claimed_amount"]))
    c2.metric("Supported", money(claim["supported_amount"]))
    c3.metric("Recommended", money(claim["recommended_amount"]))
    c4.metric("Risk", f"{claim['risk_score']}/100", claim["risk_band"])
    st.write(f"**Status:** {claim['status']}")
    st.write(f"**Patient:** {claim['patient_name']}  |  **Policy:** {claim.get('policy_no') or '—'}")
    st.write(f"**Hospital:** {claim.get('hospital') or '—'}")
    st.write(f"**Diagnosis/context:** {claim.get('diagnosis') or '—'}")

    st.subheader("Pipeline")
    stages = [
        ("Upload", True), ("OCR", bool(docs)), ("Extraction", bool(docs)),
        ("Duplicate checks", duplicates is not None), ("Clinical checks", bool(checks)),
        ("Human decision", claim["status"] in {"Approved", "Partially Approved", "Rejected", "Closed"}),
    ]
    cols = st.columns(len(stages))
    for col, (name, done) in zip(cols, stages):
        if done:
            col.success(f"✓ {name}")
        else:
            col.info(f"○ {name}")


def render_documents(docs):
    for doc in docs:
        with st.expander(f"{doc['filename']} — {doc['doc_type']} — OCR {doc['ocr_confidence']:.0%}", expanded=False):
            left, right = st.columns([1, 1])
            with left:
                path = Path(doc["stored_path"])
                if path.exists():
                    st.image(Image.open(path), use_container_width=True)
                st.caption(f"SHA-256: {doc.get('sha256', '')[:20]}… | pHash: {doc.get('phash')}")
            with right:
                st.markdown("**Reviewed transcription**")
                st.write(doc["extracted"].get("reviewed_transcription", "No reviewed transcription; inspect OCR text."))
                st.markdown("**Raw OCR text**")
                st.text_area("OCR", doc.get("ocr_text", ""), height=220, key=f"ocr_{doc['id']}")
                local_ai = (doc.get("extracted") or {}).get("local_ai") or {}
                if local_ai.get("ok"):
                    metrics = local_ai.get("metrics") or {}
                    st.success(f"Local AI extracted this document with {metrics.get('model', 'configured model')} in {metrics.get('elapsed_seconds', '—')} s.")
                    with st.expander("Local AI structured extraction"):
                        st.json(local_ai.get("data") or {})
                else:
                    st.info(f"Local AI extraction not available: {local_ai.get('error') or 'not run'}")


def render_extracted(docs):
    tests = [dict(x, source=d["filename"]) for d in docs for x in d["extracted"].get("tests", [])]
    bills = [dict(x, source=d["filename"]) for d in docs for x in d["extracted"].get("medicine_bills", [])]
    if bills:
        st.subheader("Medicine bills")
        st.dataframe(pd.DataFrame(bills), use_container_width=True, hide_index=True)
        st.write(f"**Medicine total: {money(sum(float(x.get('amount', 0)) for x in bills))}**")
    if tests:
        st.subheader("Laboratory tests / investigations")
        st.dataframe(pd.DataFrame(tests), use_container_width=True, hide_index=True)
        typed_total = sum(float(d["extracted"].get("test_total", 0) or 0) for d in docs)
        st.write(f"**Reconciled test total used by pipeline: {money(typed_total)}**")
    invoices = sorted({x for d in docs for x in d["extracted"].get("invoice_numbers", [])})
    dates = sorted({x for d in docs for x in d["extracted"].get("dates", [])})
    st.subheader("Normalized entities")
    st.write("**Invoice numbers:**", ", ".join(invoices) if invoices else "None")
    st.write("**Dates:**", ", ".join(dates) if dates else "None")
    with st.expander("Complete extracted JSON"):
        st.json({d["filename"]: d["extracted"] for d in docs})


def render_duplicates(claim, duplicates):
    st.subheader("Duplicate and multiple-application checks")
    if not duplicates:
        st.success("No duplicate match above the configured threshold.")
        st.info("Create another claim with the same sample documents to see exact and near-duplicate detection.")
        return
    rows = []
    for d in duplicates:
        rows.append({
            "Status": d["status"], "Score": d["score"], "Matched claim ID": d.get("matched_claim_id"),
            "Matched document ID": d.get("matched_document_id"), "Reasons": "; ".join(d["reasons"]),
        })
    st.dataframe(pd.DataFrame(rows), use_container_width=True, hide_index=True)
    for d in duplicates:
        with st.expander(f"{d['status']} — {d['score']:.0%}"):
            for reason in d["reasons"]:
                st.write("•", reason)


def _parse_lines(value: str) -> list[str]:
    import re as _re
    return list(dict.fromkeys(x.strip() for x in _re.split(r"[\n,;|]+", value or "") if x.strip()))


def _auto_clinical_evidence_rows(docs):
    rows = []
    for doc in docs:
        ext = doc.get("extracted") or {}
        for item in ext.get("prescribed_medicines", []) or []:
            rows.append({"Source": doc.get("filename"), "Evidence type": "Prescribed medicine", "Item": item})
        for item in ext.get("billed_medicines", []) or []:
            rows.append({"Source": doc.get("filename"), "Evidence type": "Billed medicine", "Item": item})
        for item in ext.get("tests", []) or []:
            if isinstance(item, dict) and item.get("name"):
                rows.append({"Source": doc.get("filename"), "Evidence type": "Investigation", "Item": item.get("name")})
        if ext.get("diagnosis"):
            rows.append({"Source": doc.get("filename"), "Evidence type": "Diagnosis/context", "Item": ext.get("diagnosis")})
    return rows


def render_clinical_input_form(claim, docs):
    st.subheader("Automatic local-AI evidence extraction")
    st.caption("The app reads diagnosis, prescription, itemized pharmacy bill and investigations from the uploaded images. You do not need to type them.")
    ai_status = get_ollama_status()
    c1, c2, c3 = st.columns(3)
    c1.metric("Ollama", "Connected" if ai_status["available"] else "Offline")
    c2.metric("Vision model", ai_status.get("vision_model") or "—")
    c3.metric("Model installed", "Yes" if ai_status.get("requested_model_installed") else "No")

    if not ai_status["available"]:
        st.error("Local AI is offline. Run `install_local_ai.ps1`, then return here and press the button below.")
    elif not ai_status.get("requested_model_installed"):
        st.error(f"Install the configured model first: `ollama pull {ai_status['vision_model']}`")

    force_read = st.checkbox(
        "Re-read pages already processed",
        value=False,
        key=f"force_local_ai_{claim['id']}",
        help="Leave this off to resume an interrupted run and skip completed pages.",
    )
    if st.button("Read remaining documents and refresh clinical match", type="primary", key=f"run_local_ai_{claim['id']}"):
        result = run_local_ai_with_progress(claim["id"], force=force_read)
        if result is not None:
            st.rerun()

    rows = _auto_clinical_evidence_rows(docs)
    if rows:
        st.markdown("**Automatically extracted clinical evidence**")
        st.dataframe(pd.DataFrame(rows), use_container_width=True, hide_index=True)
    else:
        st.info("No medicine-level evidence has been extracted yet. If the uploaded pharmacy page contains only bill numbers and amounts, the correct result is insufficient evidence rather than an invented medicine name.")

    with st.expander("Manual correction only — use when the image is readable but local AI made a mistake"):
        current = db.get_clinical_inputs(claim["id"])
        default_diagnosis = current.get("diagnosis_text") or claim.get("diagnosis") or ""
        default_prescribed = "\n".join(current.get("prescribed_medicines") or [])
        default_billed = "\n".join(current.get("billed_medicines") or [])
        with st.form(f"clinical_inputs_{claim['id']}"):
            diagnosis_text = st.text_area("Corrected diagnosis / treatment context", value=default_diagnosis)
            c1, c2 = st.columns(2)
            with c1:
                prescribed_text = st.text_area("Corrected prescribed medicines — one per line", value=default_prescribed, height=140)
            with c2:
                billed_text = st.text_area("Corrected itemized billed medicines — one per line", value=default_billed, height=140)
            reviewer_notes = st.text_input("Reason for correction", value=current.get("reviewer_notes") or "")
            saved = st.form_submit_button("Save correction and re-run match")
        if saved:
            db.save_clinical_inputs(claim["id"], diagnosis_text, _parse_lines(prescribed_text), _parse_lines(billed_text), reviewer_notes)
            if diagnosis_text.strip() and diagnosis_text.strip() != (claim.get("diagnosis") or "").strip():
                db.update_claim(claim["id"], diagnosis=diagnosis_text.strip())
            process_claim_compat(claim["id"], run_local_clinical=False)
            st.success("Correction saved and deterministic clinical rules refreshed.")
            st.rerun()

def render_clinical(checks):
    st.subheader("Clinical consistency review")
    st.caption(
        "The local vision model extracts evidence from the uploaded pages. Curated rules then check four independent relationships. "
        "A matched result means clinically compatible with the documented evidence; it is not claim approval."
    )

    deterministic_summary = next((c for c in checks if c.get("check_code") == "CLINICAL_MATCH_SUMMARY"), None)
    local_summary = next((c for c in checks if c.get("check_code") == "LOCAL_AI_CLINICAL_SUMMARY"), None)
    _render_overall_status(deterministic_summary, "Final evidence-gated assessment")

    dimension_order = [
        ("DIAGNOSIS_EVIDENCE_SUMMARY", "1. Diagnosis / treatment context"),
        ("DIAGNOSIS_PRESCRIPTION_SUMMARY", "2. Diagnosis ↔ prescribed medicine"),
        ("PRESCRIPTION_BILL_SUMMARY", "3. Prescribed medicine ↔ billed medicine"),
        ("DIAGNOSIS_INVESTIGATION_SUMMARY", "4. Diagnosis ↔ investigation"),
    ]
    st.markdown("#### Four-dimension result")
    fallbacks = _dimension_fallbacks(checks)
    cols = st.columns(4)
    used_fallback = False
    for col, (code, title) in zip(cols, dimension_order):
        check = next((c for c in checks if c.get("check_code") == code), None)
        if check is None:
            check = fallbacks.get(code)
            used_fallback = True
        with col:
            with st.container(border=True):
                st.markdown(f"**{title}**")
                if check:
                    _dimension_card(check)
                else:
                    st.info("Not assessed — no clinical evidence has been generated for this claim.")
    if used_fallback:
        st.info("This claim had evidence-level findings saved without the four summary records. The cards above were reconstructed automatically. Click **Refresh clinical rules** once to save the summaries permanently.")

    rows = _aggregate_clinical_rows(checks)
    st.markdown("#### Evidence-level clinical results")
    if rows:
        frame = pd.DataFrame(rows)
        preferred = [
            "Dimension", "Item", "Status", "Evidence confidence", "Occurrences",
            "Dates", "Finding", "Required evidence", "Reference", "Open reference",
        ]
        st.dataframe(
            frame[preferred],
            use_container_width=True,
            hide_index=True,
            column_config={
                "Open reference": st.column_config.LinkColumn(
                    "Open reference",
                    help="Open the clinical guidance used by the rule.",
                    display_text="Open source",
                )
            },
        )

        references = []
        seen_urls = set()
        for row in rows:
            url = str(row.get("Open reference") or "").strip()
            title = str(row.get("Reference") or "Clinical reference").strip()
            if url and url not in seen_urls:
                seen_urls.add(url)
                references.append((title, url))
        if references:
            with st.expander("Clinical reference links", expanded=False):
                for title, url in references:
                    st.markdown(f"- [{title}]({url})")
    else:
        st.info("No evidence-level medicine or investigation results are available yet.")

    information_checks = []
    for check in checks:
        code = str(check.get("check_code") or "")
        if code in SUMMARY_CODES or code.startswith("LOCAL_AI_"):
            continue
        status = _check_status(check)
        if status != "INFORMATION":
            continue
        evidence = check.get("evidence") or {}
        if not isinstance(evidence, dict):
            evidence = {}
        information_checks.append({
            "Evidence area": evidence.get("dimension") or code,
            "Item": evidence.get("item") or "—",
            "Observation": check.get("explanation") or "",
            "Confidence": _confidence_text(evidence.get("confidence")),
        })
    if information_checks:
        with st.expander("Document completeness and date observations", expanded=False):
            st.dataframe(pd.DataFrame(information_checks), use_container_width=True, hide_index=True)
            st.caption("These observations are not fraud warnings and do not indicate claim risk by themselves.")

    local_checks = [c for c in checks if str(c.get("check_code") or "").startswith("LOCAL_AI_")]
    if local_checks:
        with st.expander("Local open-source model cross-check", expanded=False):
            st.caption(
                "This is a secondary cross-check from Ollama. It cannot override the evidence-gated rule result, approve a claim, or turn a missing itemized bill into a medicine match."
            )
            _render_overall_status(local_summary, "Local-model cross-check")
            local_rows = []
            for check in local_checks:
                if check.get("check_code") in {"LOCAL_AI_CLINICAL_SUMMARY", "LOCAL_AI_UNAVAILABLE"}:
                    continue
                evidence = check.get("evidence") or {}
                if not isinstance(evidence, dict):
                    evidence = {}
                local_rows.append({
                    "Dimension": evidence.get("dimension") or "Local AI",
                    "Item": evidence.get("item") or "—",
                    "Status": _status_text(_check_status(check)),
                    "Confidence": _confidence_text(evidence.get("confidence")),
                    "Finding": check.get("explanation") or "",
                    "Required evidence": evidence.get("required_evidence") or "—",
                })
            if local_rows:
                st.dataframe(pd.DataFrame(local_rows), use_container_width=True, hide_index=True)

    st.info(
        "For the uploaded sample, pregnancy-specific scans can match antenatal care. Glucose, thyroid, vitamin B12 and vitamin D testing remain conditional unless the corresponding diagnosis, abnormal result or risk factor is documented. "
        "Medicine matching remains insufficient when the pharmacy page lists only bill numbers, dates and amounts."
    )


def render_finance(claim, docs):
    med = sum(float(d["extracted"].get("medicine_total", 0) or 0) for d in docs)
    tests = sum(float(d["extracted"].get("test_total", 0) or 0) for d in docs)
    table = pd.DataFrame([
        {"Category": "Medicine bills", "Amount": med},
        {"Category": "Laboratory tests", "Amount": tests},
        {"Category": "Total supported", "Amount": claim["supported_amount"]},
        {"Category": "Claimed", "Amount": claim["claimed_amount"]},
        {"Category": "Recommended before policy adjudication", "Amount": claim["recommended_amount"]},
    ])
    st.subheader("Financial reconciliation")
    st.dataframe(table, use_container_width=True, hide_index=True)
    gap = float(claim["claimed_amount"] or 0) - float(claim["supported_amount"] or 0)
    if gap > 0:
        st.warning(f"Unsupported or unreconciled gap: {money(gap)}")
    else:
        st.success("The structured medicine and test totals reconcile to the claimed amount.")


def render_chat(claim, docs, duplicates, checks):
    st.subheader("Claim-aware chatbot")
    st.caption("Answers are generated from the claim evidence. Any requested change is saved as a review proposal and requires approval.")
    examples = "Examples: Do the medicines match the diagnosis? What is the medicine total? Is this duplicate? List all tests. Why is it under review? Change diagnosis to hypothyroidism and antenatal care."
    st.info(examples)
    question = st.text_input("Ask about this claim")
    if st.button("Ask", type="primary") and question:
        result = answer_question(claim, docs, duplicates, checks, question)
        st.markdown(result["answer"])
        if result.get("evidence"):
            st.caption("Evidence: " + " | ".join(result["evidence"]))

    st.subheader("Correction proposals")
    proposals = db.list_corrections(claim["id"])
    if not proposals:
        st.write("No correction proposals.")
        return
    for p in proposals:
        with st.expander(f"#{p['id']} {p['field_name']} — {p['status']}"):
            st.write(f"Current: `{p['current_value']}`")
            st.write(f"Proposed: `{p['proposed_value']}`")
            st.write(f"Reason: {p['reason']}")
            if p["status"] == "Pending":
                c1, c2 = st.columns(2)
                if c1.button("Approve", key=f"approve_{p['id']}"):
                    db.review_correction(p["id"], True, "medical_officer")
                    st.rerun()
                if c2.button("Reject", key=f"reject_{p['id']}"):
                    db.review_correction(p["id"], False, "medical_officer")
                    st.rerun()


def render_audit(claim_id):
    events = db.list_audit(claim_id)
    if not events:
        st.info("No audit events.")
        return
    rows = [{"Time": x["created_at"], "Actor": x["actor"], "Action": x["action"], "Details": json.dumps(x["details"], ensure_ascii=False)} for x in events]
    st.dataframe(pd.DataFrame(rows), use_container_width=True, hide_index=True)


def render_claim_360():
    st.title("Claim 360°")
    flash = st.session_state.pop("claim_delete_flash", None)
    if flash:
        st.success(flash)
    claim_id = claim_selector("claim_360_selector")
    if claim_id is None:
        return
    claim = db.get_claim(claim_id)
    docs = db.list_documents(claim_id)
    duplicates = db.list_duplicate_matches(claim_id)
    checks = db.list_clinical_checks(claim_id)
    saved_codes = {str(c.get("check_code") or "") for c in checks}
    required_summary_codes = DIMENSION_SUMMARY_CODES | {"CLINICAL_MATCH_SUMMARY"}
    needs_clinical_refresh = (
        any(code in LEGACY_CLINICAL_CODES for code in saved_codes)
        or (bool(checks) and not required_summary_codes.issubset(saved_codes))
    )
    # Never run OCR, duplicate analysis, clinical rules, or Ollama automatically
    # while the user is merely opening Claim 360. Older saved evidence is rendered
    # immediately through the UI fallback cards. Refresh is an explicit user action.
    if needs_clinical_refresh:
        st.info(
            "Older saved clinical findings were detected. They are displayed using "
            "reconstructed summary cards. Click **Refresh clinical rules** once to "
            "persist the new summaries; local AI will not run."
        )

    c1, c2, c3 = st.columns([1, 1.7, 4.3])
    if c1.button("Refresh clinical rules"):
        with st.spinner("Refreshing deterministic clinical rules..."):
            process_claim_compat(claim_id, run_local_clinical=False)
        st.rerun()
    if c2.button("Read remaining pages with local AI", type="primary"):
        result = run_local_ai_with_progress(claim_id, force=False)
        if result is not None:
            st.rerun()
    c3.write(f"**{claim['claim_no']} — {claim['patient_name']}**")

    with st.expander("Delete this claim", expanded=False):
        _delete_claim_panel(claim_id, key_prefix=f"claim360_{claim_id}")

    tabs = st.tabs(["Summary", "Identity", "Documents & OCR", "Extracted data", "Duplicates", "Clinical checks", "Finance", "Chat & corrections", "Audit"])
    with tabs[0]:
        render_summary(claim, docs, duplicates, checks)
    with tabs[1]:
        render_identity_review(claim, docs, key_prefix=f"claim360_{claim_id}")
    with tabs[2]:
        render_documents(docs)
    with tabs[3]:
        render_extracted(docs)
    with tabs[4]:
        render_duplicates(claim, duplicates)
    with tabs[5]:
        render_clinical_input_form(claim, docs)
        st.divider()
        render_clinical(checks)
    with tabs[6]:
        render_finance(claim, docs)
    with tabs[7]:
        render_chat(claim, docs, duplicates, checks)
    with tabs[8]:
        render_audit(claim_id)


def render_duplicate_lab():
    st.title("Duplicate Lab")
    st.write("Use this page to test multiple-application detection.")
    claims = db.list_claims()
    if len(claims) < 2:
        st.info("At least two claims are needed. Load the sample, then create another claim using one or more of the same images.")
    rows = []
    for c in claims:
        for d in db.list_duplicate_matches(c["id"]):
            rows.append({"Claim": c["claim_no"], "Matched claim ID": d.get("matched_claim_id"), "Status": d["status"], "Score": d["score"], "Reasons": "; ".join(d["reasons"])})
    if rows:
        st.dataframe(pd.DataFrame(rows), use_container_width=True, hide_index=True)
    else:
        st.write("No recorded duplicate links.")


def render_settings():
    st.title("Local AI and clinical rules")
    config = load_config()
    status = get_ollama_status()
    if status["available"]:
        st.success(f"Ollama connected at {status['url']}")
    else:
        st.error(f"Ollama is not reachable at {status['url']}. Run `install_local_ai.ps1`.")
    st.write("**Installed models:**", ", ".join(status.get("models") or []) or "None detected")

    with st.form("local_ai_settings"):
        enabled = st.checkbox("Enable automatic local AI", value=bool(config.get("enabled", True)))
        ollama_url = st.text_input("Ollama URL", value=str(config.get("ollama_url") or "http://127.0.0.1:11434"))
        vision_model = st.text_input("Vision extraction model", value=str(config.get("vision_model") or "qwen2.5vl:7b"))
        clinical_model = st.text_input("Clinical matching model", value=str(config.get("clinical_model") or vision_model))
        c1, c2 = st.columns(2)
        with c1:
            timeout_seconds = st.number_input("Maximum seconds per model call", min_value=30, max_value=600, value=int(config.get("timeout_seconds") or 150), step=30)
            max_image_dimension = st.number_input("Maximum image dimension", min_value=768, max_value=2400, value=int(config.get("max_image_dimension") or 1280), step=128)
        with c2:
            clinical_attach_images = st.checkbox("Resend all images during clinical matching", value=bool(config.get("clinical_attach_images", False)), help="Usually leave off. The clinical pass can use the structured evidence already extracted from each page.")
            skip_existing_document_ai = st.checkbox("Skip pages already processed", value=bool(config.get("skip_existing_document_ai", True)))
        saved = st.form_submit_button("Save local AI settings")
    if saved:
        save_config({
            **config,
            "enabled": enabled,
            "ollama_url": ollama_url,
            "vision_model": vision_model,
            "clinical_model": clinical_model,
            "timeout_seconds": int(timeout_seconds),
            "max_image_dimension": int(max_image_dimension),
            "clinical_attach_images": clinical_attach_images,
            "skip_existing_document_ai": skip_existing_document_ai,
        })
        st.success("Local AI settings saved. Restart is not required.")
        st.rerun()

    st.code(f"ollama pull {vision_model}", language="powershell")
    st.caption("The vision model reads the original images. Structured JSON output is evidence-gated; bill amounts alone are never converted into medicine names.")

    st.subheader("Deterministic clinical rule pack")
    rules_path = Path("medclaim/data/clinical_rules.json")
    rules = json.loads(rules_path.read_text(encoding="utf-8"))
    st.json(rules)
    st.caption("The local model and deterministic rules are shown separately so a reviewer can see disagreement rather than receiving a hidden automatic decision.")

def main():
    page = st.sidebar.radio(
        "Navigation",
        ["Dashboard", "New Claim", "Claim 360", "Duplicate Lab", "Rules"],
    )
    st.sidebar.caption("Document intelligence and clinical review")

    routes = {
        "Dashboard": render_dashboard,
        "New Claim": render_new_claim,
        "Claim 360": render_claim_360,
        "Duplicate Lab": render_duplicate_lab,
        "Rules": render_settings,
    }
    routes[page]()


if __name__ == "__main__":
    main()
