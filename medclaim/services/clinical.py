from __future__ import annotations

import json
import re
from datetime import date, datetime
from pathlib import Path
from typing import Any, Iterable

from rapidfuzz import fuzz

RULES_PATH = Path("medclaim/data/clinical_rules.json")

STATUS_MATCHED = "MATCHED"
STATUS_PARTIAL = "PARTIALLY_MATCHED"
STATUS_CONDITIONAL = "CONDITIONAL_MATCH"
STATUS_MISMATCH = "MISMATCH"
STATUS_INSUFFICIENT = "INSUFFICIENT_EVIDENCE"
STATUS_MEDICAL_REVIEW = "MEDICAL_OFFICER_REVIEW"
STATUS_INFORMATION = "INFORMATION"


def load_rules() -> dict[str, Any]:
    return json.loads(RULES_PATH.read_text(encoding="utf-8"))


def _norm(value: Any) -> str:
    text = str(value or "").lower()
    text = text.replace("–", "-").replace("—", "-")
    text = re.sub(r"[^\w+./\-\s]", " ", text, flags=re.UNICODE).replace("_", " ")
    return re.sub(r"\s+", " ", text).strip()


def _split_items(value: Any) -> list[str]:
    if value is None:
        return []
    if isinstance(value, list):
        out: list[str] = []
        for item in value:
            if isinstance(item, dict):
                name = item.get("name") or item.get("medicine") or item.get("generic_name")
                if name:
                    out.append(str(name).strip())
            elif str(item).strip():
                out.append(str(item).strip())
        return list(dict.fromkeys(out))
    text = str(value)
    parts = re.split(r"[\n,;|]+", text)
    return list(dict.fromkeys(x.strip() for x in parts if x.strip()))


def _concepts_from_text(text: str, rules: dict[str, Any]) -> list[dict[str, Any]]:
    normal = _norm(text)
    found: list[dict[str, Any]] = []
    for concept in rules.get("diagnosis_concepts", []):
        matches = [kw for kw in concept.get("keywords", []) if _norm(kw) and _norm(kw) in normal]
        if matches:
            found.append({
                "code": concept["code"],
                "label": concept.get("label", concept["code"]),
                "matched_keywords": matches,
            })
    return found


def _document_diagnosis_text(documents: list[dict[str, Any]]) -> str:
    chunks: list[str] = []
    eligible_types = {
        "medical_certificate_or_prescription",
        "discharge_summary",
        "claim_form",
        "supporting_document",
    }
    for doc in documents:
        if doc.get("doc_type") not in eligible_types:
            continue
        ext = doc.get("extracted", {}) or {}
        if ext.get("diagnosis"):
            chunks.append(str(ext["diagnosis"]))
        if ext.get("pregnancy_noted"):
            chunks.append("pregnancy antenatal")
        reviewed = ext.get("reviewed_transcription")
        if reviewed:
            chunks.append(str(reviewed))
        elif doc.get("ocr_text"):
            chunks.append(str(doc["ocr_text"]))
    return "\n".join(chunks)


def _map_rule(item_name: str, rule_list: list[dict[str, Any]], alias_key: str = "aliases") -> tuple[dict[str, Any] | None, float, str | None]:
    name = _norm(item_name)
    best: tuple[dict[str, Any] | None, float, str | None] = (None, 0.0, None)
    for rule in rule_list:
        for alias in rule.get(alias_key, []):
            norm_alias = _norm(alias)
            if not norm_alias:
                continue
            if norm_alias in name or name in norm_alias:
                score = 1.0
            else:
                score = fuzz.token_set_ratio(name, norm_alias) / 100.0
            if score > best[1]:
                best = (rule, score, alias)
    if best[1] < 0.78:
        return None, best[1], best[2]
    return best



def _relationship_confidence(mapping_confidence: float, status: str) -> float:
    """Evidence confidence, not certainty that a treatment is medically necessary."""
    base = max(0.0, min(float(mapping_confidence or 0), 1.0))
    if status == STATUS_MATCHED:
        return min(0.95, max(0.70, base * 0.95))
    if status == STATUS_CONDITIONAL:
        return min(0.82, max(0.55, base * 0.82))
    if status == STATUS_MISMATCH:
        return min(0.80, max(0.50, base * 0.80))
    if status == STATUS_MEDICAL_REVIEW:
        return min(0.70, max(0.35, base * 0.70))
    if status == STATUS_INSUFFICIENT:
        return 0.95
    return min(0.90, max(0.50, base))

def _as_check(
    *,
    check_code: str,
    status: str,
    explanation: str,
    evidence: Any,
    severity: str | None = None,
    dimension: str | None = None,
    item: str | None = None,
    confidence: float | None = None,
    required_evidence: str | None = None,
    source_title: str | None = None,
    source_url: str | None = None,
) -> dict[str, Any]:
    if severity is None:
        severity = {
            STATUS_MATCHED: "low",
            STATUS_INFORMATION: "low",
            STATUS_PARTIAL: "medium",
            STATUS_CONDITIONAL: "medium",
            STATUS_INSUFFICIENT: "medium",
            STATUS_MEDICAL_REVIEW: "medium",
            STATUS_MISMATCH: "high",
        }.get(status, "medium")
    result = "consistent" if status == STATUS_MATCHED else "information" if status == STATUS_INFORMATION else "review"
    payload = {
        "clinical_status": status,
        "dimension": dimension,
        "item": item,
        "confidence": round(float(confidence), 3) if confidence is not None else None,
        "required_evidence": required_evidence,
        "source_title": source_title,
        "source_url": source_url,
        "raw_evidence": evidence,
        "engine": "deterministic_rules",
    }
    return {
        "check_code": check_code,
        "severity": severity,
        "result": result,
        "explanation": explanation,
        "evidence": payload,
    }


def _collect_tests(documents: list[dict[str, Any]]) -> list[dict[str, Any]]:
    tests: list[dict[str, Any]] = []
    for doc in documents:
        for item in doc.get("extracted", {}).get("tests", []) or []:
            if not isinstance(item, dict) or not item.get("name"):
                continue
            row = dict(item)
            row["source_document"] = doc.get("filename")
            tests.append(row)
    # Preserve distinct service lines while preventing exact OCR duplicates.
    deduped: dict[tuple[str, str, str], dict[str, Any]] = {}
    for item in tests:
        key = (_norm(item.get("name")), str(item.get("date") or ""), str(item.get("amount") or ""))
        deduped[key] = item
    return list(deduped.values())


def _collect_medicines(documents: list[dict[str, Any]], clinical_inputs: dict[str, Any]) -> tuple[list[str], list[str]]:
    prescribed = _split_items(clinical_inputs.get("prescribed_medicines"))
    billed = _split_items(clinical_inputs.get("billed_medicines"))
    for doc in documents:
        ext = doc.get("extracted", {}) or {}
        prescribed.extend(_split_items(ext.get("prescribed_medicines")))
        prescribed.extend(_split_items(ext.get("medicines")))
        billed.extend(_split_items(ext.get("billed_medicines")))
        billed.extend(_split_items(ext.get("medicine_items")))
    return list(dict.fromkeys(prescribed)), list(dict.fromkeys(billed))


def _status_for_relationship(rule: dict[str, Any], concept_codes: set[str]) -> str:
    supported = set(rule.get("supported_diagnoses", []))
    conditional = set(rule.get("conditional_diagnoses", []))
    if supported & concept_codes:
        return STATUS_MATCHED
    if conditional & concept_codes:
        return STATUS_CONDITIONAL
    if not concept_codes:
        return STATUS_INSUFFICIENT
    return STATUS_MISMATCH


def _parse_date(value: Any) -> date | None:
    text = str(value or "").strip().replace("/", "-").replace(".", "-")
    for fmt in ("%d-%m-%Y", "%d-%m-%y", "%Y-%m-%d"):
        try:
            parsed = datetime.strptime(text, fmt).date()
            if parsed.year < 2000:
                parsed = parsed.replace(year=parsed.year + 2000)
            return parsed
        except ValueError:
            continue
    return None


def _date_checks(claim: dict[str, Any], documents: list[dict[str, Any]]) -> list[dict[str, Any]]:
    service_dates: set[date] = set()
    verification_dates: set[date] = set()
    service_types = {"medicine_bill_summary", "laboratory_bill_summary", "discharge_summary"}
    claim_end = _parse_date(claim.get("treatment_end"))
    for doc in documents:
        for raw in doc.get("extracted", {}).get("dates", []) or []:
            parsed = _parse_date(raw)
            if not parsed:
                continue
            if doc.get("doc_type") in service_types:
                service_dates.add(parsed)
            elif claim_end and parsed > claim_end:
                verification_dates.add(parsed)
            else:
                # Dates repeated on a certificate can be supporting service dates.
                service_dates.add(parsed)

    checks: list[dict[str, Any]] = []
    if service_dates:
        ordered = sorted(service_dates)
        start, end = ordered[0], ordered[-1]
        explanation = f"{len(ordered)} distinct treatment/billing date(s) were detected from {start:%d-%m-%Y} to {end:%d-%m-%Y}."
        checks.append(_as_check(
            check_code="SERVICE_DATE_COVERAGE",
            status=STATUS_INFORMATION,
            explanation=explanation,
            evidence=[x.isoformat() for x in ordered],
            dimension="Date consistency",
            item="Treatment and billing dates",
            confidence=0.98,
        ))
        claim_start = _parse_date(claim.get("treatment_start"))
        claim_end = _parse_date(claim.get("treatment_end"))
        if claim_start and start < claim_start or claim_end and end > claim_end:
            checks.append(_as_check(
                check_code="DATE_OUTSIDE_TREATMENT_WINDOW",
                status=STATUS_MEDICAL_REVIEW,
                explanation="One or more billing/service dates fall outside the structured treatment period. Verify whether the treatment dates or document classification need correction.",
                evidence={"service_start": start.isoformat(), "service_end": end.isoformat(), "claim_start": str(claim.get("treatment_start") or ""), "claim_end": str(claim.get("treatment_end") or "")},
                dimension="Date consistency",
                item="Treatment window",
                confidence=0.9,
            ))
    if verification_dates:
        ordered = sorted(verification_dates)
        checks.append(_as_check(
            check_code="DOCUMENT_VERIFICATION_DATES",
            status=STATUS_INFORMATION,
            explanation="Document signing/verification dates were kept separate from treatment and billing dates: " + ", ".join(x.strftime("%d-%m-%Y") for x in ordered) + ".",
            evidence=[x.isoformat() for x in ordered],
            dimension="Date consistency",
            item="Verification dates",
            confidence=0.9,
        ))
    return checks



def _aggregate_dimension_status(statuses: list[str], *, empty_status: str = STATUS_INSUFFICIENT) -> str:
    """Return a conservative dimension-level status.

    A dimension is never called MATCHED when evidence is missing or conditional.
    Clear mismatches are escalated to medical-officer review rather than being used
    as an automatic rejection decision.
    """
    clean = [x for x in statuses if x]
    if not clean:
        return empty_status
    if STATUS_MISMATCH in clean or STATUS_MEDICAL_REVIEW in clean:
        return STATUS_MEDICAL_REVIEW
    if STATUS_INSUFFICIENT in clean:
        return STATUS_INSUFFICIENT
    if STATUS_CONDITIONAL in clean or STATUS_PARTIAL in clean:
        return STATUS_PARTIAL
    if all(x in {STATUS_MATCHED, STATUS_INFORMATION} for x in clean):
        return STATUS_MATCHED
    return STATUS_PARTIAL


def _dimension_summary_check(
    *,
    code: str,
    dimension: str,
    status: str,
    explanation: str,
    confidence: float,
    required_evidence: str | None = None,
    counts: dict[str, int] | None = None,
) -> dict[str, Any]:
    return _as_check(
        check_code=code,
        status=status,
        explanation=explanation,
        evidence={"counts": counts or {}},
        dimension=dimension,
        item="Dimension summary",
        confidence=confidence,
        required_evidence=required_evidence,
    )

def run_clinical_checks(
    claim: dict[str, Any],
    documents: list[dict[str, Any]],
    clinical_inputs: dict[str, Any] | None = None,
) -> list[dict[str, Any]]:
    """Run conservative, explainable diagnosis–medicine–test matching.

    This function deliberately distinguishes a document being present from clinical
    compatibility. Missing itemized medicine names yields INSUFFICIENT_EVIDENCE,
    never an automatic clinical match.
    """
    rules = load_rules()
    clinical_inputs = clinical_inputs or {}
    diagnosis_text = "\n".join(
        x for x in [
            str(clinical_inputs.get("diagnosis_text") or ""),
            str(claim.get("diagnosis") or ""),
            _document_diagnosis_text(documents),
        ] if x.strip()
    )
    concepts = _concepts_from_text(diagnosis_text, rules)
    concept_codes = {x["code"] for x in concepts}
    checks: list[dict[str, Any]] = []

    if concepts:
        checks.append(_as_check(
            check_code="DIAGNOSIS_CONTEXT",
            status=STATUS_INFORMATION,
            explanation="Recognized treatment context: " + ", ".join(x["label"] for x in concepts) + ".",
            evidence={"diagnosis_text": diagnosis_text, "concepts": concepts},
            dimension="Diagnosis evidence",
            item="Documented diagnosis/context",
            confidence=0.92 if claim.get("diagnosis") else 0.72,
        ))
    else:
        checks.append(_as_check(
            check_code="DIAGNOSIS_MISSING",
            status=STATUS_INSUFFICIENT,
            explanation="No sufficiently specific diagnosis or treatment context was identified. Medicine–diagnosis matching cannot be completed safely.",
            evidence={"diagnosis_text": diagnosis_text},
            dimension="Diagnosis evidence",
            item="Diagnosis/context",
            confidence=0.25,
            required_evidence="Readable diagnosis, treatment summary or medical-officer confirmation.",
        ))

    tests = _collect_tests(documents)
    for test in tests:
        name = str(test.get("name") or "")
        rule, mapping_confidence, alias = _map_rule(name, rules.get("tests", []))
        if not rule:
            checks.append(_as_check(
                check_code="TEST_UNMAPPED",
                status=STATUS_MEDICAL_REVIEW,
                explanation=f"The investigation '{name}' was extracted but is not mapped in the configured clinical dictionary.",
                evidence=test,
                dimension="Diagnosis ↔ investigation",
                item=name,
                confidence=mapping_confidence,
                required_evidence="Medical-officer review or an approved rule-dictionary update.",
            ))
            continue
        status = _status_for_relationship(rule, concept_codes)
        if status == STATUS_MATCHED:
            explanation = f"{name} matches the documented context. {rule.get('reason', '')}".strip()
        elif status == STATUS_CONDITIONAL:
            explanation = f"{name} can be related to the documented context, but the available diagnosis is not specific enough for a definitive match. {rule.get('reason', '')}".strip()
        elif status == STATUS_INSUFFICIENT:
            explanation = f"{name} was identified, but there is no sufficiently specific diagnosis against which to assess it."
        else:
            explanation = f"{name} is not directly supported by the recognized diagnosis concepts. Medical-officer review is required before treating it as inconsistent."
        checks.append(_as_check(
            check_code="TEST_DIAGNOSIS_MATCH",
            status=status,
            explanation=explanation,
            evidence={"test": test, "mapped_rule": rule.get("code"), "matched_alias": alias, "diagnosis_concepts": sorted(concept_codes)},
            dimension="Diagnosis ↔ investigation",
            item=name,
            confidence=_relationship_confidence(mapping_confidence, status),
            required_evidence="Specific diagnosis, risk factor or medical-officer note." if status != STATUS_MATCHED else None,
            source_title=rule.get("source_title"),
            source_url=rule.get("source_url"),
        ))

    prescribed, billed = _collect_medicines(documents, clinical_inputs)
    has_medicine_bill = any(d.get("doc_type") == "medicine_bill_summary" for d in documents)
    has_prescription_doc = any(d.get("doc_type") == "medical_certificate_or_prescription" for d in documents)

    checks.append(_as_check(
        check_code="PRESCRIPTION_DOCUMENT_PRESENCE",
        status=STATUS_INFORMATION,
        explanation=("A medical certificate/prescription-type document is present." if has_prescription_doc else "No medical certificate/prescription-type document was identified."),
        evidence={"present": has_prescription_doc},
        dimension="Document completeness",
        item="Prescription/medical certificate",
        confidence=0.95,
    ))

    if has_medicine_bill and not billed:
        checks.append(_as_check(
            check_code="BILL_ITEMIZATION_MISSING",
            status=STATUS_INSUFFICIENT,
            explanation="Medicine-bill totals are present, but medicine names, strength, dosage and quantity are absent. The pharmacy amount cannot be clinically matched to the diagnosis.",
            evidence={"medicine_bill_present": True, "billed_medicines": billed},
            dimension="Prescription ↔ pharmacy bill",
            item="Itemized pharmacy bill",
            confidence=0.99,
            required_evidence="Upload an itemized pharmacy invoice showing medicine name, strength, quantity and amount.",
        ))
    if not prescribed:
        checks.append(_as_check(
            check_code="PRESCRIBED_MEDICINES_MISSING",
            status=STATUS_INSUFFICIENT,
            explanation="No reliable itemized list of prescribed medicines was extracted or confirmed. A prescription document being present does not prove that the medicines match the diagnosis.",
            evidence={"prescribed_medicines": prescribed},
            dimension="Diagnosis ↔ prescribed medicine",
            item="Prescribed medicine list",
            confidence=0.99,
            required_evidence="Readable prescription or reviewer-entered prescribed medicine names.",
        ))

    mapped_prescribed: dict[str, dict[str, Any]] = {}
    for medicine in prescribed:
        rule, mapping_confidence, alias = _map_rule(medicine, rules.get("medicines", []))
        if not rule:
            checks.append(_as_check(
                check_code="MEDICINE_UNMAPPED",
                status=STATUS_MEDICAL_REVIEW,
                explanation=f"Medicine '{medicine}' could not be mapped confidently to a generic medicine in the approved dictionary.",
                evidence={"medicine": medicine},
                dimension="Diagnosis ↔ prescribed medicine",
                item=medicine,
                confidence=mapping_confidence,
                required_evidence="Confirm generic name, strength, dose and indication with a medical officer.",
            ))
            continue
        mapped_prescribed[rule["code"]] = {"raw": medicine, "rule": rule, "confidence": mapping_confidence}
        status = _status_for_relationship(rule, concept_codes)
        if status == STATUS_MATCHED:
            explanation = f"{medicine} is clinically consistent with the recognized diagnosis/context. {rule.get('reason', '')}".strip()
        elif status == STATUS_CONDITIONAL:
            explanation = f"{medicine} may be clinically related, but the diagnosis/evidence is not specific enough for a definitive match. {rule.get('reason', '')}".strip()
        elif status == STATUS_INSUFFICIENT:
            explanation = f"{medicine} was identified, but no sufficiently specific diagnosis was available."
        else:
            explanation = f"{medicine} is not supported by the recognized diagnosis concepts in the current documents. This is a review flag, not an automatic rejection."
        checks.append(_as_check(
            check_code="MEDICINE_DIAGNOSIS_MATCH",
            status=status,
            explanation=explanation,
            evidence={"medicine": medicine, "generic_name": rule.get("generic_name"), "mapped_rule": rule.get("code"), "matched_alias": alias, "diagnosis_concepts": sorted(concept_codes)},
            dimension="Diagnosis ↔ prescribed medicine",
            item=medicine,
            confidence=_relationship_confidence(mapping_confidence, status),
            required_evidence=rule.get("required_evidence") if status != STATUS_MATCHED else None,
            source_title=rule.get("source_title"),
            source_url=rule.get("source_url"),
        ))

    mapped_billed: dict[str, dict[str, Any]] = {}
    for medicine in billed:
        rule, mapping_confidence, alias = _map_rule(medicine, rules.get("medicines", []))
        if rule:
            mapped_billed[rule["code"]] = {"raw": medicine, "rule": rule, "confidence": mapping_confidence, "alias": alias}
        else:
            checks.append(_as_check(
                check_code="BILLED_MEDICINE_UNMAPPED",
                status=STATUS_MEDICAL_REVIEW,
                explanation=f"Billed medicine '{medicine}' could not be normalized to the approved medicine dictionary.",
                evidence={"medicine": medicine},
                dimension="Prescription ↔ pharmacy bill",
                item=medicine,
                confidence=mapping_confidence,
                required_evidence="Confirm the generic name from the itemized invoice.",
            ))

    if prescribed and billed:
        prescribed_codes = set(mapped_prescribed)
        billed_codes = set(mapped_billed)
        matched_codes = prescribed_codes & billed_codes
        missing_from_bill = prescribed_codes - billed_codes
        not_on_prescription = billed_codes - prescribed_codes
        for code in sorted(matched_codes):
            checks.append(_as_check(
                check_code="PRESCRIPTION_BILL_MATCH",
                status=STATUS_MATCHED,
                explanation=f"The billed item '{mapped_billed[code]['raw']}' matches prescribed medicine '{mapped_prescribed[code]['raw']}' after generic-name normalization.",
                evidence={"prescribed": mapped_prescribed[code]["raw"], "billed": mapped_billed[code]["raw"], "generic_code": code},
                dimension="Prescription ↔ pharmacy bill",
                item=mapped_prescribed[code]["rule"].get("generic_name", code),
                confidence=min(0.95, min(mapped_prescribed[code]["confidence"], mapped_billed[code]["confidence"])),
            ))
        for code in sorted(not_on_prescription):
            checks.append(_as_check(
                check_code="BILLED_NOT_PRESCRIBED",
                status=STATUS_MISMATCH,
                explanation=f"Billed medicine '{mapped_billed[code]['raw']}' was not found in the confirmed prescription list.",
                evidence={"billed": mapped_billed[code]["raw"], "generic_code": code},
                dimension="Prescription ↔ pharmacy bill",
                item=mapped_billed[code]["raw"],
                confidence=mapped_billed[code]["confidence"],
                required_evidence="Readable prescription amendment, substitution note or medical-officer confirmation.",
            ))
        for code in sorted(missing_from_bill):
            checks.append(_as_check(
                check_code="PRESCRIBED_NOT_BILLED",
                status=STATUS_INFORMATION,
                explanation=f"Prescribed medicine '{mapped_prescribed[code]['raw']}' was not found in the submitted itemized pharmacy list. This does not by itself indicate a claim problem.",
                evidence={"prescribed": mapped_prescribed[code]["raw"], "generic_code": code},
                dimension="Prescription ↔ pharmacy bill",
                item=mapped_prescribed[code]["raw"],
                confidence=mapped_prescribed[code]["confidence"],
            ))

    checks.extend(_date_checks(claim, documents))

    def statuses_for(*codes: str) -> list[str]:
        wanted = set(codes)
        return [
            str((c.get("evidence") or {}).get("clinical_status") or "")
            for c in checks
            if c.get("check_code") in wanted
        ]

    diagnosis_status = STATUS_MATCHED if concepts else STATUS_INSUFFICIENT
    diagnosis_summary = _dimension_summary_check(
        code="DIAGNOSIS_EVIDENCE_SUMMARY",
        dimension="1. Diagnosis / treatment context",
        status=diagnosis_status,
        explanation=(
            "A sufficiently specific diagnosis/treatment context was recognized from the claim evidence: "
            + ", ".join(x["label"] for x in concepts) + "."
            if concepts else
            "A sufficiently specific diagnosis/treatment context was not found, so clinical matching cannot be completed safely."
        ),
        confidence=0.92 if claim.get("diagnosis") and concepts else 0.75 if concepts else 0.95,
        required_evidence=None if concepts else "Readable diagnosis, treatment summary or medical-officer confirmation.",
    )

    medicine_statuses = statuses_for("MEDICINE_DIAGNOSIS_MATCH", "MEDICINE_UNMAPPED")
    if not prescribed:
        medicine_dimension_status = STATUS_INSUFFICIENT
        medicine_explanation = "No reliable prescribed-medicine list was extracted. Document presence alone is not a medicine–diagnosis match."
        medicine_required = "Readable prescription showing medicine name, strength, dose and duration."
    else:
        medicine_dimension_status = _aggregate_dimension_status(medicine_statuses)
        medicine_explanation = f"{len(prescribed)} prescribed medicine item(s) were assessed against the documented diagnosis/context."
        medicine_required = "Medical-officer review of conditional, unmapped or mismatched medicines." if medicine_dimension_status != STATUS_MATCHED else None
    medicine_summary = _dimension_summary_check(
        code="DIAGNOSIS_PRESCRIPTION_SUMMARY",
        dimension="2. Diagnosis ↔ prescribed medicine",
        status=medicine_dimension_status,
        explanation=medicine_explanation,
        confidence=0.98 if not prescribed else 0.86,
        required_evidence=medicine_required,
        counts={x: medicine_statuses.count(x) for x in sorted(set(medicine_statuses))},
    )

    bill_statuses = statuses_for(
        "PRESCRIPTION_BILL_MATCH", "BILLED_NOT_PRESCRIBED", "PRESCRIBED_NOT_BILLED",
        "BILL_ITEMIZATION_MISSING", "BILLED_MEDICINE_UNMAPPED"
    )
    if has_medicine_bill and not billed:
        bill_dimension_status = STATUS_INSUFFICIENT
        bill_explanation = "The pharmacy schedule contains bill/date/amount information but no itemized medicine names, so prescription-to-bill matching is not possible."
        bill_required = "Itemized pharmacy invoice showing medicine name, strength, quantity and amount."
    elif not has_medicine_bill:
        bill_dimension_status = STATUS_INSUFFICIENT
        bill_explanation = "No pharmacy-bill document was identified for prescription-to-bill matching."
        bill_required = "Itemized pharmacy invoice."
    elif not prescribed:
        bill_dimension_status = STATUS_INSUFFICIENT
        bill_explanation = "Billed medicine items cannot be matched because the prescribed-medicine list is missing."
        bill_required = "Readable prescription and itemized pharmacy invoice."
    else:
        bill_dimension_status = _aggregate_dimension_status(bill_statuses)
        bill_explanation = "Prescription and itemized billed medicines were compared after brand-to-generic normalization."
        bill_required = "Readable substitution note or medical-officer confirmation for unmatched items." if bill_dimension_status != STATUS_MATCHED else None
    bill_summary = _dimension_summary_check(
        code="PRESCRIPTION_BILL_SUMMARY",
        dimension="3. Prescribed medicine ↔ billed medicine",
        status=bill_dimension_status,
        explanation=bill_explanation,
        confidence=0.99 if has_medicine_bill and not billed else 0.86,
        required_evidence=bill_required,
        counts={x: bill_statuses.count(x) for x in sorted(set(bill_statuses))},
    )

    test_statuses = statuses_for("TEST_DIAGNOSIS_MATCH", "TEST_UNMAPPED")
    if not tests:
        test_dimension_status = STATUS_INFORMATION
        test_explanation = "No investigation/test items were submitted for this claim; this dimension is not applicable."
        test_required = None
    else:
        test_dimension_status = _aggregate_dimension_status(test_statuses)
        matched_count = test_statuses.count(STATUS_MATCHED)
        conditional_count = test_statuses.count(STATUS_CONDITIONAL)
        test_explanation = f"{len(tests)} investigation line(s) were assessed: {matched_count} matched and {conditional_count} conditional."
        test_required = "Specific diagnosis/risk factor for conditional tests." if test_dimension_status != STATUS_MATCHED else None
    test_summary = _dimension_summary_check(
        code="DIAGNOSIS_INVESTIGATION_SUMMARY",
        dimension="4. Diagnosis ↔ investigation",
        status=test_dimension_status,
        explanation=test_explanation,
        confidence=0.90 if tests else 0.98,
        required_evidence=test_required,
        counts={x: test_statuses.count(x) for x in sorted(set(test_statuses))},
    )

    dimension_statuses = [
        x for x in [diagnosis_status, medicine_dimension_status, bill_dimension_status, test_dimension_status]
        if x != STATUS_INFORMATION
    ]
    if STATUS_MEDICAL_REVIEW in dimension_statuses or STATUS_MISMATCH in dimension_statuses:
        overall = STATUS_MEDICAL_REVIEW
        overall_explanation = "At least one clinical dimension requires medical-officer review. No automatic approval or rejection is made."
    elif STATUS_INSUFFICIENT in dimension_statuses:
        overall = STATUS_INSUFFICIENT
        overall_explanation = "Investigation evidence can be reviewed, but medicine-level matching is incomplete because prescribed and/or itemized billed medicine evidence is missing."
    elif STATUS_PARTIAL in dimension_statuses or STATUS_CONDITIONAL in dimension_statuses:
        overall = STATUS_PARTIAL
        overall_explanation = "The evidence is partly consistent, but one or more clinical relationships are conditional and need supporting context."
    else:
        overall = STATUS_MATCHED
        overall_explanation = "All four evidence dimensions are clinically compatible within the configured decision-support rules."

    substantive = [
        c for c in checks
        if c["check_code"] not in {
            "DIAGNOSIS_CONTEXT", "PRESCRIPTION_DOCUMENT_PRESENCE",
            "SERVICE_DATE_COVERAGE", "DOCUMENT_VERIFICATION_DATES"
        }
    ]
    statuses = [str((c.get("evidence") or {}).get("clinical_status") or "") for c in substantive]
    summary_evidence = {
        "clinical_status": overall,
        "diagnosis_concepts": concepts,
        "prescribed_medicines": prescribed,
        "billed_medicines": billed,
        "test_count": len(tests),
        "dimension_statuses": {
            "diagnosis": diagnosis_status,
            "diagnosis_prescription": medicine_dimension_status,
            "prescription_bill": bill_dimension_status,
            "diagnosis_investigation": test_dimension_status,
        },
        "counts": {status: statuses.count(status) for status in sorted(set(statuses))},
        "disclaimer": rules.get("disclaimer"),
    }
    summary = _as_check(
        check_code="CLINICAL_MATCH_SUMMARY",
        status=overall,
        explanation=overall_explanation,
        evidence=summary_evidence,
        dimension="Overall clinical assessment",
        item="Diagnosis–prescription–bill–investigation consistency",
        confidence=0.95 if overall == STATUS_INSUFFICIENT else 0.86,
        required_evidence="Readable prescription and itemized pharmacy invoice." if overall == STATUS_INSUFFICIENT else None,
    )
    return [summary, diagnosis_summary, medicine_summary, bill_summary, test_summary] + checks
