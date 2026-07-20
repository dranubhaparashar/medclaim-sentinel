from medclaim.services.risk import calculate_risk


def test_risk_explains_duplicate_and_gap():
    risk = calculate_risk(
        [{"score": 0.9}],
        [{"result": "review", "severity": "high"}],
        [{"ocr_confidence": 0.2}],
        20000,
        10000,
    )
    assert risk["score"] >= 50
    assert risk["reasons"]
