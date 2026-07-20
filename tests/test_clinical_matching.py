import json
from pathlib import Path

from medclaim.services.clinical import run_clinical_checks


def _documents_with_medicine_bill():
    return [
        {
            "filename": "prescription.jpg",
            "doc_type": "medical_certificate_or_prescription",
            "ocr_text": "",
            "extracted": {"dates": ["01-01-25"]},
        },
        {
            "filename": "pharmacy.jpg",
            "doc_type": "medicine_bill_summary",
            "ocr_text": "",
            "extracted": {"medicine_total": 1000, "dates": ["01-01-25"]},
        },
    ]


def _status(check):
    return check["evidence"]["clinical_status"]


def test_sample_requires_itemized_medicine_evidence():
    config = json.loads(Path("sample-data/sample_claim_transcription.json").read_text(encoding="utf-8"))
    documents = [
        {
            "filename": filename,
            "doc_type": override["doc_type"],
            "ocr_text": override.get("reviewed_text", ""),
            "extracted": override.get("extracted", {}),
        }
        for filename, override in config["documents"].items()
    ]
    checks = run_clinical_checks(config["claim"], documents, {})
    summary = next(x for x in checks if x["check_code"] == "CLINICAL_MATCH_SUMMARY")
    assert _status(summary) == "INSUFFICIENT_EVIDENCE"
    assert any(x["check_code"] == "BILL_ITEMIZATION_MISSING" for x in checks)
    assert not any(x["check_code"] == "MEDICINE_DIAGNOSIS_MATCH" for x in checks)


def test_folic_acid_matches_antenatal_diagnosis_and_bill():
    checks = run_clinical_checks(
        {"diagnosis": "Pregnancy / antenatal care"},
        _documents_with_medicine_bill(),
        {
            "prescribed_medicines": ["Folic acid 5 mg"],
            "billed_medicines": ["Folvite 5 mg"],
        },
    )
    med = next(x for x in checks if x["check_code"] == "MEDICINE_DIAGNOSIS_MATCH")
    bill = next(x for x in checks if x["check_code"] == "PRESCRIPTION_BILL_MATCH")
    summary = next(x for x in checks if x["check_code"] == "CLINICAL_MATCH_SUMMARY")
    assert _status(med) == "MATCHED"
    assert _status(bill) == "MATCHED"
    assert _status(summary) == "MATCHED"


def test_levothyroxine_is_only_conditional_for_pregnancy_without_thyroid_diagnosis():
    checks = run_clinical_checks(
        {"diagnosis": "Pregnancy / antenatal care"},
        _documents_with_medicine_bill(),
        {
            "prescribed_medicines": ["Levothyroxine 50 mcg"],
            "billed_medicines": ["Thyronorm 50 mcg"],
        },
    )
    med = next(x for x in checks if x["check_code"] == "MEDICINE_DIAGNOSIS_MATCH")
    assert _status(med) == "CONDITIONAL_MATCH"


def test_levothyroxine_matches_documented_hypothyroidism():
    checks = run_clinical_checks(
        {"diagnosis": "Pregnancy with hypothyroidism"},
        _documents_with_medicine_bill(),
        {
            "prescribed_medicines": ["Levothyroxine 50 mcg"],
            "billed_medicines": ["Thyronorm 50 mcg"],
        },
    )
    med = next(x for x in checks if x["check_code"] == "MEDICINE_DIAGNOSIS_MATCH")
    assert _status(med) == "MATCHED"


def test_billed_medicine_not_on_prescription_is_flagged():
    checks = run_clinical_checks(
        {"diagnosis": "Pregnancy / antenatal care"},
        _documents_with_medicine_bill(),
        {
            "prescribed_medicines": ["Folic acid 5 mg"],
            "billed_medicines": ["Folvite 5 mg", "Metformin 500 mg"],
        },
    )
    mismatch = next(x for x in checks if x["check_code"] == "BILLED_NOT_PRESCRIBED")
    summary = next(x for x in checks if x["check_code"] == "CLINICAL_MATCH_SUMMARY")
    assert _status(mismatch) == "MISMATCH"
    assert _status(summary) == "MEDICAL_OFFICER_REVIEW"
