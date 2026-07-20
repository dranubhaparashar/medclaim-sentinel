from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Any

from medclaim.services.identity import resolve_bundle_identity

DATE_RE = re.compile(r"\b(?:0?[1-9]|[12]\d|3[01])[-/.](?:0?[1-9]|1[0-2])[-/.](?:\d{2}|\d{4})\b")
INVOICE_RE = re.compile(r"\bA?0?\d{5,7}\b", re.I)
AMOUNT_RE = re.compile(r"(?<!\d)(?:₹\s*)?(\d{2,6})(?:\.\d{1,2})?(?!\d)")
TOTAL_RE = re.compile(r"(?:grand\s+total|total|कुल|योग)\s*[:\-]?\s*(?:₹\s*)?(\d{2,7})(?:\.\d{1,2})?", re.I)

PATIENT_PATTERNS = [
    re.compile(r"(?:patient\s*(?:name)?|name\s+of\s+(?:the\s+)?patient|insured\s*name)\s*[:\-]?\s*([A-Za-z][A-Za-z .]{2,60})", re.I),
    re.compile(r"(?:रोगी(?:\s+का)?\s+नाम|नाम)\s*[:\-]?\s*([^\n]{2,60})", re.I),
]
POLICY_PATTERNS = [
    re.compile(r"(?:policy|member|uhid|health\s*id)\s*(?:no\.?|number)?\s*[:\-]?\s*([A-Z0-9\-/]{4,30})", re.I),
]
APPLICATION_PATTERNS = [
    re.compile(r"(?:application|claim)\s*(?:no\.?|number)\s*[:\-]?\s*([A-Z0-9\-/]{3,30})", re.I),
]

CLINICAL_RULES_PATH = Path("medclaim/data/clinical_rules.json")
DOSAGE_CUE_RE = re.compile(r"\b(?:tab(?:let)?|cap(?:sule)?|syrup|inj(?:ection)?|mcg|mg|ml|od|bd|tds|hs|daily|once|twice)\b", re.I)
LAB_CUE_RE = re.compile(r"\b(?:test|profile|serum|plasma|level|assay|hba1c|tsh|t3|t4|cbc|fbs|ppbs|urine)\b", re.I)


def _medicine_aliases() -> list[tuple[str, str]]:
    try:
        rules = json.loads(CLINICAL_RULES_PATH.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return []
    aliases: list[tuple[str, str]] = []
    for rule in rules.get("medicines", []):
        generic = str(rule.get("generic_name") or rule.get("code") or "").strip()
        for alias in rule.get("aliases", []):
            aliases.append((str(alias).lower(), generic))
    return aliases


def extract_medicine_mentions(text: str, document_type: str) -> dict[str, list[str]]:
    """Conservatively extract medicine names only when a medicine/prescription cue exists.

    Lab names such as 'thyroxine test' are intentionally excluded so that a thyroid
    investigation is not mistaken for levothyroxine medication.
    """
    if document_type not in {"medical_certificate_or_prescription", "medicine_bill_summary", "supporting_document"}:
        return {"prescribed_medicines": [], "billed_medicines": []}
    found: list[str] = []
    for raw_line in text.splitlines():
        line = raw_line.strip()
        low = line.lower()
        if not line or LAB_CUE_RE.search(line):
            continue
        for alias, generic in _medicine_aliases():
            if alias in low and (DOSAGE_CUE_RE.search(line) or document_type == "medicine_bill_summary"):
                found.append(generic)
    found = list(dict.fromkeys(found))
    if document_type == "medicine_bill_summary":
        return {"prescribed_medicines": [], "billed_medicines": found}
    return {"prescribed_medicines": found, "billed_medicines": []}


def normalize_date(value: str) -> str:
    return value.replace("/", "-").replace(".", "-")


def classify_document(text: str, filename: str) -> str:
    t = f"{filename} {text}".lower()
    if "detail of medicine bill" in t or "medical store" in t:
        return "medicine_bill_summary"
    if "detail of test" in t or "name of lab" in t or "urine routine" in t:
        return "laboratory_bill_summary"
    if "discharge" in t:
        return "discharge_summary"
    if "prescription" in t or "दवाइयों के नाम" in t or "चिकित्सा अधिकारी" in t:
        return "medical_certificate_or_prescription"
    if "claim" in t or "reimbursement" in t or "प्रतिपूर्ति" in t:
        return "claim_form"
    return "supporting_document"


def _clean_candidate(value: str) -> str:
    value = re.sub(r"\s+", " ", value).strip(" :-.,|_")
    return value[:120]


def extract_generic(text: str) -> dict[str, Any]:
    dates = sorted(set(normalize_date(x) for x in DATE_RE.findall(text)))
    invoices = sorted(set(x.upper() for x in INVOICE_RE.findall(text) if any(c.isdigit() for c in x)))
    amounts = []
    for match in AMOUNT_RE.findall(text):
        try:
            n = int(match)
        except ValueError:
            continue
        if 20 <= n <= 1_000_000:
            amounts.append(n)
    explicit_totals = []
    for match in TOTAL_RE.findall(text):
        try:
            explicit_totals.append(float(match))
        except ValueError:
            pass
    return {
        "dates": dates,
        "invoice_numbers": invoices,
        "amount_candidates": amounts[:100],
        "total_candidate": max(explicit_totals) if explicit_totals else (max(amounts) if amounts else None),
        "explicit_total_candidates": explicit_totals,
    }


def extract_document(text: str, filename: str) -> dict[str, Any]:
    result = extract_generic(text)
    document_type = classify_document(text, filename)
    result["document_type"] = document_type
    result.update(extract_medicine_mentions(text, document_type))
    return result


def _first_match(patterns: list[re.Pattern[str]], text: str) -> str | None:
    for pattern in patterns:
        match = pattern.search(text)
        if match:
            value = _clean_candidate(match.group(1))
            if value:
                return value
    return None


def _hospital_from_text(text: str) -> str | None:
    for raw_line in text.splitlines():
        line = _clean_candidate(raw_line)
        if not line:
            continue
        low = line.lower()
        if any(token in low for token in ("hospital", "healthcare", "medical centre", "medical center", "चिकित्सालय", "अस्पताल")):
            # Avoid returning the form header when a more specific provider line exists.
            if len(line) >= 8:
                return line
    return None


def infer_claim_metadata(documents: list[dict[str, Any]]) -> dict[str, Any]:
    """Infer optional claim metadata after OCR.

    The upload must never be blocked because these fields are missing. Values here are
    suggestions and should be reviewed by a human in Claim 360.
    """
    chunks: list[str] = []
    explicit_totals: list[float] = []
    reviewed_providers: list[str] = []
    reviewed_diagnoses: list[str] = []
    application_numbers: list[str] = []
    policy_numbers: list[str] = []
    for doc in documents:
        ext = doc.get("extracted", {}) or {}
        chunks.append(str(ext.get("reviewed_transcription") or doc.get("ocr_text") or ""))
        provider = ext.get("provider")
        if isinstance(provider, str) and provider.strip():
            reviewed_providers.append(provider.strip())
        diagnosis = ext.get("diagnosis")
        if isinstance(diagnosis, str) and diagnosis.strip():
            reviewed_diagnoses.append(diagnosis.strip())
        application_numbers.extend(str(x).strip() for x in ext.get("application_numbers", []) if str(x).strip())
        policy_numbers.extend(str(x).strip() for x in ext.get("policy_numbers", []) if str(x).strip())
        explicit_totals.extend(float(x) for x in ext.get("explicit_total_candidates", []) if isinstance(x, (int, float)))
        for key in ("medicine_total", "test_total"):
            value = ext.get(key)
            if isinstance(value, (int, float)) and value > 0:
                explicit_totals.append(float(value))
    text = "\n".join(chunks)

    metadata: dict[str, Any] = {}
    identity = resolve_bundle_identity(documents)
    if identity.get("patient_name"):
        metadata["patient_name"] = identity["patient_name"]
        metadata["patient_name_confidence"] = identity.get("confidence", 0.0)
    metadata["identity_resolution"] = identity

    # Backward-compatible labelled-text fallback when the bundle resolver has no answer.
    if not metadata.get("patient_name"):
        patient = _first_match(PATIENT_PATTERNS, text)
        if patient and not any(x in patient.lower() for x in ("hospital", "office", "doctor", "medical")):
            metadata["patient_name"] = patient
            metadata["patient_name_confidence"] = 0.55
    if application_numbers:
        metadata["application_no"] = application_numbers[0]
    else:
        application = _first_match(APPLICATION_PATTERNS, text)
        if application:
            metadata["application_no"] = application
    if policy_numbers:
        metadata["policy_no"] = policy_numbers[0]
    else:
        policy = _first_match(POLICY_PATTERNS, text)
        if policy:
            metadata["policy_no"] = policy
    if reviewed_providers:
        metadata["hospital"] = " / ".join(dict.fromkeys(reviewed_providers))
    else:
        hospital = _hospital_from_text(text)
        if hospital:
            metadata["hospital"] = hospital

    if reviewed_diagnoses:
        metadata["diagnosis"] = " / ".join(dict.fromkeys(reviewed_diagnoses))
    else:
        lower = text.lower()
        if "pregnancy" in lower or "antenatal" in lower or "गर्भ" in text:
            metadata["diagnosis"] = "Pregnancy / antenatal care (OCR-inferred; verify)"

    if explicit_totals:
        # Different attachment-category totals are additive. Exact duplicate totals are counted once.
        unique_totals = sorted({round(x, 2) for x in explicit_totals if x > 0})
        metadata["claimed_amount"] = float(sum(unique_totals))
    return metadata


def flatten_claim_data(documents: list[dict[str, Any]]) -> dict[str, Any]:
    invoices: set[str] = set()
    dates: set[str] = set()
    amounts: list[float] = []
    tests: list[dict[str, Any]] = []
    medicine_bills: list[dict[str, Any]] = []
    for doc in documents:
        ext = doc.get("extracted", {})
        invoices.update(ext.get("invoice_numbers", []))
        dates.update(ext.get("dates", []))
        amounts.extend(float(x) for x in ext.get("amount_candidates", []) if isinstance(x, (int, float)))
        tests.extend(ext.get("tests", []))
        medicine_bills.extend(ext.get("medicine_bills", []))
    return {
        "invoice_numbers": sorted(invoices),
        "dates": sorted(dates),
        "amount_candidates": amounts,
        "tests": tests,
        "medicine_bills": medicine_bills,
    }
