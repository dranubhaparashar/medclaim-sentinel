from medclaim.services.clinical import run_clinical_checks
from medclaim.services.local_ai import _postprocess_local_clinical_result, local_clinical_result_to_checks


def _status(check):
    return check["evidence"]["clinical_status"]


def test_pregnancy_specific_scan_matches_but_glucose_is_conditional():
    documents = [{
        "filename": "tests.jpg",
        "doc_type": "laboratory_bill_summary",
        "ocr_text": "",
        "extracted": {
            "tests": [
                {"name": "USG-NT/NB SCAN", "date": "03-10-24", "amount": 1200},
                {"name": "FBS", "date": "19-01-25", "amount": 50},
                {"name": "HbA1c", "date": "19-01-25", "amount": 131},
            ],
            "dates": ["03-10-24", "19-01-25"],
        },
    }]
    checks = run_clinical_checks({"diagnosis": "Pregnancy / antenatal care"}, documents, {})
    items = {c["evidence"].get("item"): _status(c) for c in checks if c["check_code"] == "TEST_DIAGNOSIS_MATCH"}
    assert items["USG-NT/NB SCAN"] == "MATCHED"
    assert items["FBS"] == "CONDITIONAL_MATCH"
    assert items["HbA1c"] == "CONDITIONAL_MATCH"
    dimension = next(c for c in checks if c["check_code"] == "DIAGNOSIS_INVESTIGATION_SUMMARY")
    assert _status(dimension) == "PARTIALLY_MATCHED"


def test_document_presence_is_information_not_clinical_match():
    documents = [{
        "filename": "certificate.jpg",
        "doc_type": "medical_certificate_or_prescription",
        "ocr_text": "pregnancy",
        "extracted": {},
    }]
    checks = run_clinical_checks({"diagnosis": "Pregnancy / antenatal care"}, documents, {})
    presence = next(c for c in checks if c["check_code"] == "PRESCRIPTION_DOCUMENT_PRESENCE")
    assert _status(presence) == "INFORMATION"
    medicine_dimension = next(c for c in checks if c["check_code"] == "DIAGNOSIS_PRESCRIPTION_SUMMARY")
    assert _status(medicine_dimension) == "INSUFFICIENT_EVIDENCE"


def test_deterministic_checks_have_evidence_confidence():
    documents = [{
        "filename": "tests.jpg",
        "doc_type": "laboratory_bill_summary",
        "ocr_text": "",
        "extracted": {"tests": [{"name": "CBC", "date": "19-01-25", "amount": 250}]},
    }]
    checks = run_clinical_checks({"diagnosis": "Pregnancy / antenatal care"}, documents, {})
    for check in checks:
        evidence = check.get("evidence") or {}
        assert evidence.get("confidence") is not None


def test_local_model_cannot_mark_glucose_matched_from_pregnancy_alone():
    raw = {
        "overall_status": "MATCHED",
        "overall_explanation": "all matched",
        "diagnoses_used": ["pregnancy"],
        "medicine_matches": [],
        "investigation_matches": [{
            "investigation": "HbA1c",
            "status": "MATCHED",
            "matched_diagnosis": "pregnancy",
            "rationale": "screening",
            "required_evidence": "",
            "confidence": 0.95,
        }],
        "prescription_bill_matches": [],
        "missing_evidence": [],
        "model_warning": "",
    }
    evidence = {
        "diagnoses": ["Pregnancy / antenatal care"],
        "prescribed_medicines": [],
        "billed_medicines": [],
        "itemized_medicine_names_present": False,
    }
    fixed = _postprocess_local_clinical_result(raw, evidence)
    assert fixed["investigation_matches"][0]["status"] == "CONDITIONAL_MATCH"
    assert fixed["overall_status"] == "INSUFFICIENT_EVIDENCE"
    checks = local_clinical_result_to_checks({"ok": True, "data": fixed, "metrics": {}})
    assert checks[0]["evidence"]["confidence"] is not None
