from __future__ import annotations

import re
from typing import Any

from medclaim import db


def _money(value: float | int | None) -> str:
    return f"₹{float(value or 0):,.0f}"


def answer_question(claim: dict[str, Any], documents: list[dict[str, Any]], duplicates: list[dict[str, Any]],
                    clinical_checks: list[dict[str, Any]], question: str) -> dict[str, Any]:
    q = question.strip().lower()
    evidence: list[str] = []

    correction = re.search(r"(?:change|correct|update)\s+([a-z_ ]+?)\s+(?:from\s+(.+?)\s+)?to\s+(.+)$", q)
    if correction:
        field = correction.group(1).strip().replace(" ", "_")
        aliases = {
            "patient": "patient_name", "patient_name": "patient_name", "policy": "policy_no",
            "policy_number": "policy_no", "diagnosis": "diagnosis", "hospital": "hospital",
            "claimed_amount": "claimed_amount", "supported_amount": "supported_amount",
            "recommended_amount": "recommended_amount", "status": "status",
        }
        field = aliases.get(field, field)
        allowed = set(aliases.values())
        if field not in allowed:
            return {"answer": f"I cannot directly propose a change to '{field}'. Use a supported claim field or correct the OCR field in the document review panel.", "evidence": [], "proposal_id": None}
        current = claim.get(field)
        proposed = correction.group(3).strip().strip('"\'')
        proposal_id = db.propose_correction(claim["id"], field, current, proposed, "Requested through claim chat")
        return {
            "answer": f"I created correction proposal #{proposal_id}: {field} from '{current}' to '{proposed}'. It has not changed the claim yet; a reviewer must approve it.",
            "evidence": ["Correction proposal and audit history"],
            "proposal_id": proposal_id,
        }

    if any(k in q for k in ["medicine total", "total medicine", "medical bill total"]):
        medicine_total = sum(float(x.get("amount", 0)) for d in documents for x in d.get("extracted", {}).get("medicine_bills", []))
        if medicine_total == 0:
            medicine_total = sum(float(d.get("extracted", {}).get("medicine_total", 0) or 0) for d in documents)
        evidence = [d["filename"] for d in documents if d.get("doc_type") == "medicine_bill_summary"]
        return {"answer": f"The extracted medicine-bill total is {_money(medicine_total)}.", "evidence": evidence}

    if any(k in q for k in ["test total", "laboratory total", "lab total"]):
        test_total = sum(float(x.get("amount", 0)) for d in documents for x in d.get("extracted", {}).get("tests", []))
        if test_total == 0:
            test_total = sum(float(d.get("extracted", {}).get("test_total", 0) or 0) for d in documents)
        evidence = [d["filename"] for d in documents if d.get("doc_type") == "laboratory_bill_summary"]
        return {"answer": f"The extracted laboratory-test total is {_money(test_total)}.", "evidence": evidence}

    if "total" in q or "amount" in q:
        return {
            "answer": f"Claimed: {_money(claim.get('claimed_amount'))}; supported: {_money(claim.get('supported_amount'))}; recommended: {_money(claim.get('recommended_amount'))}.",
            "evidence": ["Claim financial reconciliation"],
        }

    if any(phrase in q for phrase in [
        "medicine match", "medicines match", "medicine matches", "diagnosis match",
        "clinically match", "clinical match", "medicine appropriate", "medicine consistent"
    ]):
        # The deterministic evidence-gated result is the primary answer. The local
        # model is a secondary cross-check and never overrides missing evidence.
        summary = next((c for c in clinical_checks if c.get("check_code") == "CLINICAL_MATCH_SUMMARY"), None)
        local_summary = next((c for c in clinical_checks if c.get("check_code") == "LOCAL_AI_CLINICAL_SUMMARY"), None)
        medicine_checks = [c for c in clinical_checks if c.get("check_code") == "MEDICINE_DIAGNOSIS_MATCH" or str(c.get("check_code", "")).startswith("LOCAL_AI_MEDICINE_MATCH_")]
        bill_checks = [c for c in clinical_checks if c.get("check_code") in {"PRESCRIPTION_BILL_MATCH", "BILLED_NOT_PRESCRIBED", "BILL_ITEMIZATION_MISSING", "PRESCRIBED_MEDICINES_MISSING"} or str(c.get("check_code", "")).startswith("LOCAL_AI_PRESCRIPTION_BILL_MATCH_")]
        if not summary:
            return {"answer": "The clinical matching pipeline has not produced a summary yet. Re-run the claim pipeline.", "evidence": ["Clinical checks"]}
        payload = summary.get("evidence") or {}
        status = payload.get("clinical_status", "REVIEW")
        answer = f"Final evidence-gated clinical status: {status}. {summary.get('explanation', '')}"
        if local_summary:
            local_payload = local_summary.get("evidence") or {}
            local_status = local_payload.get("clinical_status", "REVIEW")
            if local_status != status:
                answer += f" The local-model cross-check returned {local_status}; it is shown separately and does not override the final evidence-gated status."
        if medicine_checks:
            details = []
            for check in medicine_checks:
                ev = check.get("evidence") or {}
                details.append(f"{ev.get('item') or 'medicine'}: {ev.get('clinical_status') or check.get('result')} — {check.get('explanation')}")
            answer += " Medicine findings: " + " | ".join(details)
        elif bill_checks:
            answer += " " + " ".join(c.get("explanation", "") for c in bill_checks[:2])
        return {"answer": answer, "evidence": ["Clinical Match Summary", "Diagnosis–medicine checks", "Prescription–bill checks"]}

    if "duplicate" in q or "submitted more than" in q or "multiple claim" in q:
        if not duplicates:
            return {"answer": "No duplicate match is currently recorded for this claim.", "evidence": ["Duplicate-check results"]}
        best = duplicates[0]
        return {
            "answer": f"The strongest duplicate result is '{best['status']}' with a score of {best['score']:.0%}. Reasons: " + "; ".join(best["reasons"]),
            "evidence": [f"Matched claim database ID {best.get('matched_claim_id')}", "Duplicate-check results"],
        }

    if "test" in q or "investigation" in q:
        tests = [x for d in documents for x in d.get("extracted", {}).get("tests", [])]
        if not tests:
            return {"answer": "No structured test entries were extracted. Review the OCR text and low-confidence fields.", "evidence": ["Document extraction"]}
        text = ", ".join(f"{x['name']} ({_money(x.get('amount'))}, {x.get('date', 'date unavailable')})" for x in tests)
        return {"answer": f"Extracted tests: {text}.", "evidence": ["Laboratory bill summary"]}

    if "why" in q and ("review" in q or "risk" in q):
        flags = [c["explanation"] for c in clinical_checks if c["result"] == "review"]
        dup = [r for x in duplicates for r in x["reasons"]]
        reasons = dup + flags
        return {"answer": "Review reasons: " + ("; ".join(reasons) if reasons else "No material automated review flag."), "evidence": ["Risk and consistency checks"]}

    if "missing" in q or "document" in q:
        types = {d.get("doc_type") for d in documents}
        expected = {"medical_certificate_or_prescription", "medicine_bill_summary", "laboratory_bill_summary"}
        missing = sorted(expected - types)
        return {"answer": "Missing expected document categories: " + (", ".join(missing) if missing else "none in the MVP checklist"), "evidence": ["Document classification"]}

    if "status" in q:
        return {"answer": f"The claim status is '{claim.get('status')}'. Risk band: {claim.get('risk_band')} ({claim.get('risk_score')}/100).", "evidence": ["Claim header"]}

    if "invoice" in q or "bill number" in q:
        invoices = sorted({x for d in documents for x in d.get("extracted", {}).get("invoice_numbers", [])})
        return {"answer": "Extracted invoice numbers: " + (", ".join(invoices) if invoices else "none"), "evidence": ["Structured document extraction"]}

    return {
        "answer": "I can answer questions about totals, tests, diagnosis–medicine matching, prescription–bill matching, invoice numbers, duplicate matches, missing documents, risk reasons and claim status. I can also create a controlled correction proposal, for example: 'change diagnosis to hypothyroidism and antenatal care'.",
        "evidence": ["Current claim record"],
    }
