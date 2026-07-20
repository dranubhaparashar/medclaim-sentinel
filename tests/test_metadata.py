from medclaim.services.extract import infer_claim_metadata


def test_infers_metadata_without_manual_form_fields():
    docs = [{
        "ocr_text": "Patient Name: Anu Sharma\nSPARSH MULTI SPECIALITY HOSPITAL, SHIMLA\nDiagnosis: Pregnancy\nTOTAL 11694",
        "extracted": {"explicit_total_candidates": [11694]},
    }]
    result = infer_claim_metadata(docs)
    assert result["patient_name"] == "Anu Sharma"
    assert "HOSPITAL" in result["hospital"]
    assert result["claimed_amount"] == 11694
    assert "Pregnancy" in result["diagnosis"]
