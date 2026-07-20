from __future__ import annotations

import base64
import json
import os
import re
import urllib.error
import urllib.request
from pathlib import Path
from typing import Any

from PIL import Image, ImageEnhance, ImageFilter, ImageOps

# These exclusions prevent provider names and form labels from being mistaken for patients.
EXCLUDED_NAME_TERMS = {
    "hospital", "healthcare", "medical", "store", "shimla", "office", "doctor",
    "patient", "claim", "detail", "total", "sparsh", "atulaya", "speciality",
    "amount", "date", "dealer", "test", "medicine", "government", "accountant",
    "रुपये", "प्राप्त", "परामर्श", "उपचार", "प्रमाणित", "कार्यालय", "प्रदेश", "परीक्षा", "सेवा",
}

LATIN_NAME_RE = re.compile(r"\b([A-Z][a-z]{2,24}(?:\s+[A-Z][a-z]{2,24}){1,3})\b")
DEVANAGARI_NAME_RE = re.compile(r"([\u0900-\u097F]{2,20}(?:\s+[\u0900-\u097F]{2,20}){1,3})")


def _clean_name(value: str) -> str:
    value = re.sub(r"\s+", " ", value).strip(" .,:;|_-\n\t")
    return value[:100]


def _looks_like_name(value: str) -> bool:
    cleaned = _clean_name(value)
    if len(cleaned) < 5 or len(cleaned) > 80:
        return False
    low = cleaned.lower()
    if any(term in low for term in EXCLUDED_NAME_TERMS):
        return False
    tokens = cleaned.split()
    if len(tokens) < 2 or len(tokens) > 4:
        return False
    if sum(ch.isdigit() for ch in cleaned) > 0:
        return False
    return True


def candidates_from_text(text: str) -> list[dict[str, Any]]:
    candidates: list[dict[str, Any]] = []
    seen: set[str] = set()

    # Prefer text found after common identity labels.
    label_patterns = [
        re.compile(r"(?:patient\s*(?:name)?|insured\s*name|claimant\s*name)\s*[:\-]?\s*([^\n]{4,80})", re.I),
        re.compile(r"(?:रोगी(?:\s+का)?\s+नाम|नाम)\s*[:\-]?\s*([^\n]{4,80})", re.I),
        re.compile(r"(?:श्री|श्रीमती|सुश्री|कु\.?|कुमारी)\s*[:\-]?\s*([^\n]{4,80})", re.I),
    ]
    for pattern in label_patterns:
        for match in pattern.finditer(text):
            raw = _clean_name(match.group(1))
            # Stop when a relation/employment label begins.
            raw = re.split(r"\b(?:son|daughter|wife|husband|office|department)\b|(?:पुत्र|पुत्री|पत्नी|पति|कार्यालय)", raw, maxsplit=1, flags=re.I)[0]
            if _looks_like_name(raw):
                key = raw.casefold()
                if key not in seen:
                    seen.add(key)
                    candidates.append({"value": raw, "confidence": 0.72, "source": "labelled OCR text"})

    # Secondary candidates from ordinary title-cased or Devanagari lines.
    for regex, base_conf in ((LATIN_NAME_RE, 0.52),):
        for match in regex.finditer(text):
            raw = _clean_name(match.group(1))
            if _looks_like_name(raw):
                key = raw.casefold()
                if key not in seen:
                    seen.add(key)
                    candidates.append({"value": raw, "confidence": base_conf, "source": "unlabelled OCR text"})
    return candidates[:20]


def create_identity_crops(path: str | Path, output_dir: str | Path) -> list[dict[str, str]]:
    """Create reviewer-friendly regions likely to contain a claimant name or signature.

    The method is intentionally generic. It does not assert that a signature is the patient name;
    it only exposes relevant evidence for confirmation or a local vision model.
    """
    path = Path(path)
    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)
    image = Image.open(path).convert("RGB")
    w, h = image.size

    regions = [
        ("upper_identity", (0, int(h * 0.07), w, int(h * 0.27))),
        ("middle_identity", (0, int(h * 0.12), w, int(h * 0.36))),
        ("right_signature", (int(w * 0.48), int(h * 0.25), w, int(h * 0.52))),
        ("lower_signature", (int(w * 0.42), int(h * 0.31), w, int(h * 0.64))),
    ]

    result: list[dict[str, str]] = []
    for label, box in regions:
        crop = image.crop(box)
        # Skip almost blank crops.
        gray = ImageOps.grayscale(crop)
        extrema = gray.getextrema()
        if not extrema or extrema[1] - extrema[0] < 15:
            continue
        crop = crop.resize((max(900, crop.width * 2), max(280, crop.height * 2)))
        crop = ImageEnhance.Contrast(crop).enhance(1.35)
        crop = ImageEnhance.Sharpness(crop).enhance(1.8)
        crop = crop.filter(ImageFilter.UnsharpMask(radius=1.2, percent=130, threshold=3))
        target = out / f"{path.stem}_{label}.png"
        crop.save(target)
        result.append({"label": label, "path": str(target)})
    return result


def _extract_json_object(text: str) -> dict[str, Any] | None:
    text = text.strip()
    if text.startswith("```"):
        text = re.sub(r"^```(?:json)?", "", text).strip()
        text = re.sub(r"```$", "", text).strip()
    start = text.find("{")
    end = text.rfind("}")
    if start < 0 or end <= start:
        return None
    try:
        value = json.loads(text[start : end + 1])
    except json.JSONDecodeError:
        return None
    return value if isinstance(value, dict) else None


def ollama_identity_from_images(image_paths: list[str | Path]) -> dict[str, Any] | None:
    """Ask an optional local Ollama vision model to read the claimant name.

    No cloud or paid API is used. When Ollama is unavailable the function returns None.
    The default model can be changed with OLLAMA_VISION_MODEL.
    """
    enabled = os.getenv("ENABLE_OLLAMA_VISION", "0").strip().lower() in {"1", "true", "yes", "on"}
    if not enabled or not image_paths:
        return None

    model = os.getenv("OLLAMA_VISION_MODEL", "").strip()
    if not model:
        return None
    url = os.getenv("OLLAMA_URL", "http://127.0.0.1:11434/api/generate")
    images: list[str] = []
    for p in image_paths[:8]:
        try:
            images.append(base64.b64encode(Path(p).read_bytes()).decode("ascii"))
        except OSError:
            continue
    if not images:
        return None

    prompt = (
        "You are reading medical insurance claim documents. Identify the patient or claimant name only when visible. "
        "Use all pages together. A handwritten signature is supporting evidence but may be the claimant rather than the patient. "
        "Return strict JSON with keys patient_name, confidence (0 to 1), evidence, and needs_human_review. "
        "Do not invent a name. If uncertain, patient_name must be null and needs_human_review true."
    )
    payload = json.dumps({"model": model, "prompt": prompt, "images": images, "stream": False}).encode("utf-8")
    request = urllib.request.Request(url, data=payload, headers={"Content-Type": "application/json"}, method="POST")
    try:
        with urllib.request.urlopen(request, timeout=120) as response:
            outer = json.loads(response.read().decode("utf-8"))
    except (urllib.error.URLError, TimeoutError, json.JSONDecodeError, OSError):
        return None
    parsed = _extract_json_object(str(outer.get("response", "")))
    if not parsed:
        return None
    name = parsed.get("patient_name")
    if name is not None:
        name = _clean_name(str(name))
        if not _looks_like_name(name):
            name = None
    try:
        confidence = max(0.0, min(1.0, float(parsed.get("confidence", 0))))
    except (TypeError, ValueError):
        confidence = 0.0
    return {
        "patient_name": name,
        "confidence": confidence,
        "evidence": str(parsed.get("evidence", "Local vision model")),
        "needs_human_review": bool(parsed.get("needs_human_review", confidence < 0.8)),
        "source": f"local Ollama vision model ({model})",
    }


def resolve_bundle_identity(documents: list[dict[str, Any]]) -> dict[str, Any]:
    """Resolve identity from text candidates and an optional local vision model."""
    scored: dict[str, dict[str, Any]] = {}
    image_paths: list[str] = []
    evidence_crops: list[dict[str, str]] = []

    for doc in documents:
        ext = doc.get("extracted", {}) or {}
        text = str(ext.get("reviewed_transcription") or doc.get("ocr_text") or "")
        for candidate in ext.get("identity_candidates", []) or candidates_from_text(text):
            value = _clean_name(str(candidate.get("value", "")))
            if not _looks_like_name(value):
                continue
            key = value.casefold()
            confidence = float(candidate.get("confidence", 0.4) or 0.4)
            current = scored.setdefault(key, {"value": value, "score": 0.0, "sources": []})
            current["score"] += confidence
            current["sources"].append(str(candidate.get("source", doc.get("filename", "document"))))
        stored = doc.get("stored_path")
        if stored:
            image_paths.append(str(stored))
        for crop in ext.get("identity_crops", []) or []:
            if isinstance(crop, dict) and crop.get("path"):
                evidence_crops.append(crop)

    local_vision = ollama_identity_from_images(image_paths)
    if local_vision and local_vision.get("patient_name"):
        key = str(local_vision["patient_name"]).casefold()
        current = scored.setdefault(key, {"value": local_vision["patient_name"], "score": 0.0, "sources": []})
        current["score"] += 1.4 * float(local_vision.get("confidence", 0))
        current["sources"].append(str(local_vision.get("source")))

    ranked = sorted(scored.values(), key=lambda x: x["score"], reverse=True)
    best = ranked[0] if ranked else None
    # A name requires corroboration across pages, a high-confidence local vision result,
    # or a reviewed bundled-demo transcription.
    reviewed_source = bool(best and any("reviewed bundled sample" in source.lower() for source in best.get("sources", [])))
    accepted = bool(best and (best["score"] >= 1.15 or (reviewed_source and best["score"] >= 0.9)))
    return {
        "patient_name": best["value"] if accepted else None,
        "confidence": min(0.99, best["score"] / 1.8) if best else 0.0,
        "candidates": ranked[:10],
        "local_vision": local_vision,
        "evidence_crops": evidence_crops[:12],
        "needs_human_review": not accepted,
    }
