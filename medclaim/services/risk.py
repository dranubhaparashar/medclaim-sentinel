from __future__ import annotations

from typing import Any


def calculate_risk(
    duplicate_matches: list[dict[str, Any]],
    clinical_checks: list[dict[str, Any]],
    documents: list[dict[str, Any]],
    claimed_amount: float,
    supported_amount: float,
) -> dict[str, Any]:
    points = 0
    reasons: list[str] = []

    max_dup = max((float(x["score"]) for x in duplicate_matches), default=0.0)
    dup_points = round(40 * max_dup)
    if dup_points:
        points += dup_points
        reasons.append(f"Duplicate evidence: +{dup_points}")

    # Clinical uncertainty is kept separate from fraud. Missing itemization should
    # cause an information request, not a large fraud score.
    status_points = {
        "MISMATCH": 8,
        "MEDICAL_OFFICER_REVIEW": 5,
        "INSUFFICIENT_EVIDENCE": 2,
        "CONDITIONAL_MATCH": 1,
        "PARTIALLY_MATCHED": 1,
    }
    clinical_points = 0
    seen: set[tuple[str, str]] = set()
    for check in clinical_checks:
        evidence = check.get("evidence") or {}
        status = str(evidence.get("clinical_status") or "")
        item = str(evidence.get("item") or check.get("check_code") or "")
        key = (status or str(check.get("result") or ""), item)
        if key in seen:
            continue
        seen.add(key)
        if status:
            clinical_points += status_points.get(status, 0)
        elif check.get("result") == "review":
            # Backward compatibility for older stored checks and tests.
            clinical_points += {"high": 8, "medium": 4, "low": 1}.get(str(check.get("severity") or "medium"), 2)
    clinical_points = min(20, clinical_points)
    if clinical_points:
        points += clinical_points
        reasons.append(f"Clinical/document review needs: +{clinical_points}")

    low_conf = [d for d in documents if float(d.get("ocr_confidence") or 0) < 0.45]
    if low_conf:
        p = min(10, 2 * len(low_conf))
        points += p
        reasons.append(f"Low OCR confidence: +{p}")

    if claimed_amount > 0 and supported_amount >= 0 and supported_amount < claimed_amount:
        gap = claimed_amount - supported_amount
        ratio = gap / claimed_amount
        p = min(15, max(3, round(15 * ratio)))
        points += p
        reasons.append(f"Unsupported amount gap ₹{gap:,.0f}: +{p}")

    points = min(100, points)
    band = "high" if points >= 65 else "medium" if points >= 35 else "low"
    recommendation = "Medical/fraud review" if band == "high" else "Claim officer review" if band == "medium" else "Standard review"
    return {
        "score": points,
        "band": band,
        "recommendation": recommendation,
        "reasons": reasons or ["No material automated risk indicators"],
    }
