from __future__ import annotations

from typing import Any

import imagehash
from rapidfuzz.fuzz import ratio


def phash_similarity(a: str | None, b: str | None) -> float:
    if not a or not b:
        return 0.0
    distance = imagehash.hex_to_hash(a) - imagehash.hex_to_hash(b)
    return max(0.0, 1.0 - distance / 64.0)


def claim_duplicate_score(current_claim: dict[str, Any], current_docs: list[dict[str, Any]],
                          other_claim: dict[str, Any], other_docs: list[dict[str, Any]]) -> dict[str, Any]:
    score = 0.0
    reasons: list[str] = []

    name_sim = ratio((current_claim.get("patient_name") or "").lower(), (other_claim.get("patient_name") or "").lower()) / 100.0
    if name_sim >= 0.9:
        score += 0.18
        reasons.append("Patient names are highly similar")

    if current_claim.get("policy_no") and current_claim.get("policy_no") == other_claim.get("policy_no"):
        score += 0.12
        reasons.append("Same policy/member number")

    amount_a = float(current_claim.get("claimed_amount") or 0)
    amount_b = float(other_claim.get("claimed_amount") or 0)
    if amount_a and amount_b and abs(amount_a - amount_b) <= max(1, 0.01 * max(amount_a, amount_b)):
        score += 0.12
        reasons.append("Claimed amounts match")

    exact_doc = False
    best_phash = 0.0
    for a in current_docs:
        for b in other_docs:
            if a.get("sha256") and a.get("sha256") == b.get("sha256"):
                exact_doc = True
            best_phash = max(best_phash, phash_similarity(a.get("phash"), b.get("phash")))
    if exact_doc:
        score += 0.46
        reasons.append("An identical document file exists in both claims")
    elif best_phash >= 0.94:
        score += 0.34
        reasons.append(f"Near-identical document image ({best_phash:.0%} similarity)")

    current_invoices = {x for d in current_docs for x in d.get("extracted", {}).get("invoice_numbers", [])}
    other_invoices = {x for d in other_docs for x in d.get("extracted", {}).get("invoice_numbers", [])}
    overlap = current_invoices & other_invoices
    if overlap:
        fraction = len(overlap) / max(1, min(len(current_invoices), len(other_invoices)))
        score += min(0.34, 0.14 + 0.20 * fraction)
        reasons.append(f"{len(overlap)} invoice number(s) overlap: {', '.join(sorted(overlap)[:5])}")

    score = min(1.0, score)
    status = "probable_duplicate" if score >= 0.72 else "possible_duplicate" if score >= 0.45 else "low_similarity"
    return {"score": round(score, 3), "status": status, "reasons": reasons or ["No strong duplicate evidence"]}


def find_document_duplicates(claim_id: int, current_docs: list[dict[str, Any]], all_docs: list[dict[str, Any]]) -> list[dict[str, Any]]:
    matches = []
    for d in current_docs:
        for other in all_docs:
            if other["claim_id"] == claim_id:
                continue
            reasons = []
            score = 0.0
            if d.get("sha256") and d.get("sha256") == other.get("sha256"):
                score = 1.0
                reasons.append("Exact file SHA-256 match")
            else:
                similarity = phash_similarity(d.get("phash"), other.get("phash"))
                if similarity >= 0.90:
                    score = similarity
                    reasons.append(f"Perceptual image similarity {similarity:.0%}")
            if score >= 0.90:
                matches.append({
                    "matched_claim_id": other["claim_id"],
                    "matched_document_id": other["id"],
                    "score": round(score, 3),
                    "status": "exact_duplicate" if score == 1.0 else "near_duplicate",
                    "reasons": reasons,
                })
    matches.sort(key=lambda x: x["score"], reverse=True)
    return matches
