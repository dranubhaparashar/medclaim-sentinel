from __future__ import annotations

import json
import shutil
from pathlib import Path
from typing import Any

from medclaim import db
from medclaim.services.clinical import load_rules, run_clinical_checks
from medclaim.services.duplicate import claim_duplicate_score, find_document_duplicates
from medclaim.services.extract import extract_document, infer_claim_metadata
from medclaim.services.ocr import file_sha256, perceptual_hash, run_ocr
from medclaim.services.local_ai import (
    extract_document_with_local_ai,
    local_clinical_result_to_checks,
    merge_local_ai_into_extracted,
    run_local_clinical_match,
)
from medclaim.services.risk import calculate_risk

SAMPLE_OVERRIDE = Path("sample-data/sample_claim_transcription.json")


def save_uploaded_file(claim_id: int, filename: str, content: bytes) -> Path:
    folder = Path("storage/claims") / str(claim_id)
    folder.mkdir(parents=True, exist_ok=True)
    safe = Path(filename).name.replace(" ", "_")
    path = folder / safe
    path.write_bytes(content)
    return path


def _sample_override_for(filename: str, path: Path | None = None) -> dict[str, Any] | None:
    """Return reviewed demo metadata by filename or exact file fingerprint.

    Fingerprint matching makes the included sample behave consistently when the same
    images are uploaded with names such as 1.jpeg, 2.jpeg, etc. It is deliberately
    restricted to the bundled reviewed demo files and is not used for arbitrary claims.
    """
    if not SAMPLE_OVERRIDE.exists():
        return None
    config = json.loads(SAMPLE_OVERRIDE.read_text(encoding="utf-8"))
    documents = config.get("documents", {})
    if filename in documents:
        return documents[filename]
    if path and path.exists():
        digest = file_sha256(path)
        for sample_filename, override in documents.items():
            sample_path = Path("sample-data") / sample_filename
            if sample_path.exists() and file_sha256(sample_path) == digest:
                return override
    return None


def process_document(claim_id: int, path: Path, display_filename: str | None = None, use_reviewed_sample: bool = False) -> int:
    filename = display_filename or path.name
    override = _sample_override_for(filename, path)
    if use_reviewed_sample and override:
        ocr = {"text": override.get("reviewed_text", ""), "confidence": 0.99, "boxes": []}
    else:
        ocr = run_ocr(path)
    extracted = extract_document(ocr["text"], filename)
    extracted["identity_candidates"] = ocr.get("identity_candidates", [])
    extracted["identity_crops"] = ocr.get("identity_crops", [])
    extracted["highlighted_numbers"] = ocr.get("highlighted_numbers", [])

    # A local vision-language model reads the actual image and returns evidence-backed
    # structured fields. If Ollama is unavailable, conventional OCR remains usable and
    # the error is stored in the document rather than blocking intake.
    if use_reviewed_sample:
        local_result = {"ok": False, "error": "Reviewed sample transcription used; local image extraction skipped.", "data": {}, "metrics": {}}
    else:
        local_result = extract_document_with_local_ai(path, filename=filename, ocr_text=ocr.get("text", ""))
    extracted = merge_local_ai_into_extracted(extracted, local_result)

    # Highlighted total cells are considerably more reliable than unrestricted OCR
    # for photocopied bill-summary tables. Keep the evidence and apply it only to
    # the matching document category.
    highlighted_values = [
        int(item["value"]) for item in ocr.get("highlighted_numbers", [])
        if isinstance(item, dict) and str(item.get("value", "")).isdigit()
    ]
    if highlighted_values:
        best_total = max(highlighted_values)
        if extracted["document_type"] == "medicine_bill_summary":
            extracted["medicine_total"] = best_total
        elif extracted["document_type"] == "laboratory_bill_summary":
            extracted["test_total"] = best_total

    if override:
        extracted.update(override.get("extracted", {}))
        doc_type = override.get("doc_type", extracted.get("document_type", "supporting_document"))
        if override.get("reviewed_text"):
            extracted["reviewed_transcription"] = override["reviewed_text"]
    else:
        doc_type = extracted.get("document_type", "supporting_document")
    return db.add_document(claim_id, {
        "filename": filename,
        "stored_path": str(path),
        "sha256": file_sha256(path),
        "phash": perceptual_hash(path),
        "doc_type": doc_type,
        "ocr_text": ocr["text"],
        "ocr_confidence": ocr["confidence"],
        "extracted": extracted,
    })


def process_claim(claim_id: int, *, run_local_clinical: bool = False) -> dict[str, Any]:
    claim = db.get_claim(claim_id)
    if not claim:
        raise ValueError("Claim not found")
    documents = db.list_documents(claim_id)

    # Uploaded documents are the primary input. Infer metadata after OCR instead of
    # forcing the intake user to type information already present in the documents.
    inferred = infer_claim_metadata(documents)
    metadata_updates: dict[str, Any] = {}
    if (not claim.get("patient_name") or claim.get("patient_name") == "Pending OCR Review") and inferred.get("patient_name"):
        metadata_updates["patient_name"] = inferred["patient_name"]
    for field in ("application_no", "policy_no", "hospital", "diagnosis"):
        if not claim.get(field) and inferred.get(field):
            metadata_updates[field] = inferred[field]
    if float(claim.get("claimed_amount") or 0) <= 0 and float(inferred.get("claimed_amount") or 0) > 0:
        metadata_updates["claimed_amount"] = float(inferred["claimed_amount"])
    if metadata_updates:
        db.update_claim(claim_id, **metadata_updates)
        claim = db.get_claim(claim_id) or claim

    all_claims = db.list_claims()
    all_docs = db.list_all_documents()

    matches = find_document_duplicates(claim_id, documents, all_docs)
    for other_claim in all_claims:
        if other_claim["id"] == claim_id:
            continue
        other_docs = [d for d in all_docs if d["claim_id"] == other_claim["id"]]
        m = claim_duplicate_score(claim, documents, other_claim, other_docs)
        if m["score"] >= 0.45:
            m["matched_claim_id"] = other_claim["id"]
            m["matched_document_id"] = None
            matches.append(m)
    deduped = {}
    for m in matches:
        key = (m.get("matched_claim_id"), m.get("matched_document_id"), m["status"])
        if key not in deduped or m["score"] > deduped[key]["score"]:
            deduped[key] = m
    matches = sorted(deduped.values(), key=lambda x: x["score"], reverse=True)
    db.replace_duplicate_matches(claim_id, matches)

    clinical_inputs = db.get_clinical_inputs(claim_id)
    deterministic_checks = run_clinical_checks(claim, documents, clinical_inputs)
    checks = list(deterministic_checks)

    # Local AI is an optional cross-check. Re-running deterministic rules must be fast
    # and must not silently start another long model call. When local AI is not being
    # run, preserve only previously generated LOCAL_AI_* cross-checks and discard all
    # legacy v4/v5 check codes.
    if run_local_clinical:
        local_clinical = run_local_clinical_match(claim, documents, load_rules())
        checks.extend(local_clinical_result_to_checks(local_clinical))
        db.audit(claim_id, "local_ai", "LOCAL_AI_CLINICAL_MATCH", {
            "ok": bool(local_clinical.get("ok")),
            "error": local_clinical.get("error") or "",
            "metrics": local_clinical.get("metrics") or {},
            "overall_status": (local_clinical.get("data") or {}).get("overall_status"),
        })
    else:
        previous_local_checks = [
            c for c in db.list_clinical_checks(claim_id)
            if str(c.get("check_code") or "").startswith("LOCAL_AI_")
        ]
        checks.extend(previous_local_checks)

    db.replace_clinical_checks(claim_id, checks)

    medicine_total = sum(float(d.get("extracted", {}).get("medicine_total", 0) or 0) for d in documents)
    test_total = sum(float(d.get("extracted", {}).get("test_total", 0) or 0) for d in documents)
    supported = medicine_total + test_total
    if supported == 0:
        # Conservative fallback: do not sum arbitrary OCR amount candidates.
        supported = float(claim.get("claimed_amount") or 0)

    risk = calculate_risk(matches, deterministic_checks, documents, float(claim.get("claimed_amount") or 0), supported)
    recommended = min(float(claim.get("claimed_amount") or supported), supported)
    deterministic_summary = next(
        (c for c in deterministic_checks if c.get("check_code") == "CLINICAL_MATCH_SUMMARY"),
        None,
    )
    overall_clinical_status = str(
        ((deterministic_summary or {}).get("evidence") or {}).get("clinical_status") or "INSUFFICIENT_EVIDENCE"
    )
    if any(m["score"] >= 0.72 for m in matches):
        status = "Possible Duplicate"
    elif overall_clinical_status in {"MISMATCH", "MEDICAL_OFFICER_REVIEW"}:
        status = "Medical Review Required"
    elif overall_clinical_status in {"INSUFFICIENT_EVIDENCE", "PARTIALLY_MATCHED", "CONDITIONAL_MATCH"}:
        status = "Additional Information Required"
    else:
        status = "Ready for Assessment"
    db.update_claim(
        claim_id,
        supported_amount=supported,
        recommended_amount=recommended,
        risk_score=risk["score"],
        risk_band=risk["band"],
        status=status,
    )
    db.audit(claim_id, "system", "PIPELINE_COMPLETED", {"documents": len(documents), "duplicates": len(matches), "clinical_checks": len(checks), "risk": risk})
    return {"supported_amount": supported, "recommended_amount": recommended, "risk": risk, "status": status}



def run_local_ai_for_claim(
    claim_id: int,
    *,
    force: bool = False,
    progress_callback: Any | None = None,
) -> dict[str, Any]:
    """Re-read saved claim images with the configured local Ollama VLM.

    Pages that already have a successful local-AI extraction are skipped by default,
    so an interrupted run can resume instead of starting again. The final clinical
    pass uses the extracted structured evidence and does not resend every image unless
    ``clinical_attach_images`` is explicitly enabled in ``local_ai_config.json``.
    """
    from medclaim.services.local_ai import load_config

    documents = db.list_documents(claim_id)
    config = load_config()
    skip_existing = bool(config.get("skip_existing_document_ai", True)) and not force
    requested_model = str(config.get("vision_model") or "")
    processed = 0
    skipped = 0
    failures: list[dict[str, str]] = []
    total = len(documents)

    for index, doc in enumerate(documents, start=1):
        filename = str(doc.get("filename") or f"document-{index}")
        existing_local = (doc.get("extracted") or {}).get("local_ai") or {}
        existing_model = str(existing_local.get("processed_at_model") or (existing_local.get("metrics") or {}).get("model") or "")
        already_done = bool(existing_local.get("ok")) and (not requested_model or not existing_model or existing_model == requested_model)
        if skip_existing and already_done:
            skipped += 1
            if progress_callback:
                progress_callback(index, total, filename, "skipped")
            continue

        path = Path(str(doc.get("stored_path") or ""))
        if not path.exists():
            failures.append({"filename": filename, "error": "Stored file not found"})
            if progress_callback:
                progress_callback(index, total, filename, "failed")
            continue

        if progress_callback:
            progress_callback(index - 1, total, filename, "processing")
        result = extract_document_with_local_ai(
            path,
            filename=filename,
            ocr_text=str(doc.get("ocr_text") or ""),
        )
        merged = merge_local_ai_into_extracted(doc.get("extracted") or {}, result)
        doc_type = str(merged.get("document_type") or doc.get("doc_type") or "supporting_document")
        db.update_document_analysis(int(doc["id"]), doc_type=doc_type, extracted=merged)
        if result.get("ok"):
            processed += 1
            state = "completed"
        else:
            failures.append({"filename": filename, "error": str(result.get("error") or "Unknown local AI error")})
            state = "failed"
        if progress_callback:
            progress_callback(index, total, filename, state)

    if progress_callback:
        progress_callback(total, total, "Clinical matching", "clinical")
    pipeline_result = process_claim(claim_id, run_local_clinical=True)
    db.audit(
        claim_id,
        "local_ai",
        "CLAIM_LOCAL_AI_REANALYZED",
        {"processed": processed, "skipped": skipped, "failures": failures, "force": force},
    )
    return {"processed": processed, "skipped": skipped, "failures": failures, "pipeline": pipeline_result}

def load_sample_claim() -> int:
    config = json.loads(SAMPLE_OVERRIDE.read_text(encoding="utf-8"))
    sample = config["claim"]
    existing = db.get_claim_by_no(sample["claim_no"])
    if existing:
        return int(existing["id"])
    claim_id = db.create_claim(sample)
    folder = Path("storage/claims") / str(claim_id)
    folder.mkdir(parents=True, exist_ok=True)
    for filename in config["documents"]:
        src = Path("sample-data") / filename
        dst = folder / filename
        shutil.copy2(src, dst)
        process_document(claim_id, dst, filename, use_reviewed_sample=True)
    process_claim(claim_id)
    return claim_id
