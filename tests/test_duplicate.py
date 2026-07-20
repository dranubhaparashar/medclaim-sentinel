from medclaim.services.duplicate import claim_duplicate_score


def test_exact_document_match_is_high_risk():
    a = {"patient_name": "Asha Sharma", "policy_no": "P1", "claimed_amount": 1000}
    b = {"patient_name": "Asha Sharma", "policy_no": "P1", "claimed_amount": 1000}
    docs_a = [{"sha256": "abc", "phash": None, "extracted": {"invoice_numbers": ["A001"]}}]
    docs_b = [{"sha256": "abc", "phash": None, "extracted": {"invoice_numbers": ["A001"]}}]
    result = claim_duplicate_score(a, docs_a, b, docs_b)
    assert result["score"] >= 0.72
    assert result["status"] == "probable_duplicate"
