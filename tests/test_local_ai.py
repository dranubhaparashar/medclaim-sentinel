from medclaim.services.local_ai import local_clinical_result_to_checks, merge_local_ai_into_extracted


def test_bill_summary_does_not_invent_billed_medicines():
    result = {
        "ok": True,
        "metrics": {"model": "test-model"},
        "data": {
            "document_type": "medicine_bill_summary",
            "patient_names": [],
            "providers": ["Example Medical Store"],
            "diagnoses": [],
            "prescribed_medicines": [],
            "billed_medicines": [
                {
                    "name": "Invented Drug",
                    "strength": "500 mg",
                    "quantity": "1",
                    "amount": 500,
                    "confidence": 0.99,
                    "source_text": "bill amount only",
                }
            ],
            "tests": [],
            "bills": [{"bill_no": "A001", "date": "01-01-2025", "amount": 500, "dealer": "Example", "confidence": 0.9, "source_text": "A001 01-01-2025 500"}],
            "totals": {"medicine_total": 500, "test_total": 0, "grand_total": 500},
            "dates": ["01-01-2025"],
            "application_numbers": [],
            "policy_numbers": [],
            "itemized_medicine_names_present": False,
            "evidence_quality": "clear",
            "warnings": [],
        },
    }
    merged = merge_local_ai_into_extracted({"document_type": "medicine_bill_summary"}, result)
    assert merged.get("billed_medicines", []) == []
    assert merged["medicine_total"] == 500
    assert merged["medicine_bills"][0]["invoice_no"] == "A001"


def test_itemized_visible_medicine_is_merged():
    result = {
        "ok": True,
        "metrics": {},
        "data": {
            "document_type": "medicine_bill_summary",
            "patient_names": [],
            "providers": [],
            "diagnoses": [],
            "prescribed_medicines": [],
            "billed_medicines": [{"name": "Thyronorm", "strength": "50 mcg", "quantity": "30", "amount": 120, "confidence": 0.92, "source_text": "Thyronorm 50 mcg"}],
            "tests": [],
            "bills": [],
            "totals": {"medicine_total": 120, "test_total": 0, "grand_total": 120},
            "dates": [],
            "application_numbers": [],
            "policy_numbers": [],
            "itemized_medicine_names_present": True,
            "evidence_quality": "clear",
            "warnings": [],
        },
    }
    merged = merge_local_ai_into_extracted({}, result)
    assert merged["billed_medicines"] == ["Thyronorm 50 mcg"]


def test_local_clinical_output_becomes_visible_checks():
    checks = local_clinical_result_to_checks({
        "ok": True,
        "metrics": {"model": "test"},
        "data": {
            "overall_status": "MATCHED",
            "overall_explanation": "The evidenced medicine matches the documented diagnosis.",
            "diagnoses_used": ["hypothyroidism"],
            "medicine_matches": [{
                "medicine": "Thyronorm 50 mcg",
                "source": "prescription",
                "normalized_name": "levothyroxine",
                "status": "MATCHED",
                "matched_diagnosis": "hypothyroidism",
                "rationale": "Levothyroxine is used for hypothyroidism.",
                "required_evidence": "",
                "confidence": 0.9,
            }],
            "investigation_matches": [],
            "prescription_bill_matches": [],
            "missing_evidence": [],
            "model_warning": "Decision support only",
        },
    })
    assert checks[0]["check_code"] == "LOCAL_AI_CLINICAL_SUMMARY"
    assert checks[0]["evidence"]["clinical_status"] == "MATCHED"
    assert checks[1]["check_code"].startswith("LOCAL_AI_MEDICINE_MATCH_")
