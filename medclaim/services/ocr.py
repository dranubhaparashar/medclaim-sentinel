from __future__ import annotations

import hashlib
import os
import re
from pathlib import Path
from typing import Any

import cv2
import imagehash
import numpy as np
import pytesseract
from PIL import Image, ImageEnhance, ImageOps

from medclaim.services.identity import candidates_from_text, create_identity_crops


def configure_tesseract() -> None:
    explicit = os.getenv("TESSERACT_CMD")
    windows_default = Path(r"C:\Program Files\Tesseract-OCR\tesseract.exe")
    if explicit:
        pytesseract.pytesseract.tesseract_cmd = explicit
    elif windows_default.exists():
        pytesseract.pytesseract.tesseract_cmd = str(windows_default)


def file_sha256(path: str | Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def perceptual_hash(path: str | Path) -> str:
    return str(imagehash.phash(Image.open(path).convert("RGB")))


def _deskew(gray: np.ndarray) -> np.ndarray:
    coords = np.column_stack(np.where(gray < 245))
    if len(coords) < 100:
        return gray
    angle = cv2.minAreaRect(coords.astype(np.float32))[-1]
    angle = -(90 + angle) if angle < -45 else -angle
    if abs(angle) > 7:
        return gray
    h, w = gray.shape[:2]
    matrix = cv2.getRotationMatrix2D((w / 2, h / 2), angle, 1.0)
    return cv2.warpAffine(gray, matrix, (w, h), flags=cv2.INTER_CUBIC, borderMode=cv2.BORDER_REPLICATE)


def preprocess_image(path: str | Path) -> Image.Image:
    raw = cv2.imread(str(path))
    if raw is None:
        raise ValueError(f"Unable to read image: {path}")
    gray = cv2.cvtColor(raw, cv2.COLOR_BGR2GRAY)
    gray = cv2.fastNlMeansDenoising(gray, None, 10, 7, 21)
    gray = _deskew(gray)
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    gray = clahe.apply(gray)
    return Image.fromarray(gray)


def _tesseract_text(image: Image.Image, language: str, psm: int, extra: str = "") -> str:
    config = f"--oem 3 --psm {psm} {extra}".strip()
    try:
        return pytesseract.image_to_string(image, lang=language, config=config).strip()
    except pytesseract.TesseractError:
        return pytesseract.image_to_string(image, lang="eng", config=config).strip()


def _ocr_identity_regions(path: str | Path, full_text: str) -> tuple[list[dict[str, Any]], list[dict[str, str]]]:
    """Create evidence crops and reuse full-page OCR candidates.

    Extra Tesseract passes on cursive signatures are both slow and usually inaccurate.
    A local vision model can inspect the crops when enabled; otherwise the reviewer sees
    the same crops in Claim 360 and confirms the identity without re-uploading.
    """
    path = Path(path)
    crop_dir = path.parent / "identity_evidence"
    crops = create_identity_crops(path, crop_dir)
    return candidates_from_text(full_text), crops

def _yellow_highlighted_numbers(path: str | Path) -> list[dict[str, Any]]:
    """Read numbers from yellow-highlighted cells, commonly used for bill totals."""
    raw = cv2.imread(str(path))
    if raw is None:
        return []
    hsv = cv2.cvtColor(raw, cv2.COLOR_BGR2HSV)
    # Broad threshold handles pale photocopied yellow and stronger marker highlights.
    mask = cv2.inRange(hsv, np.array([12, 25, 105]), np.array([50, 255, 255]))
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, np.ones((7, 17), np.uint8), iterations=2)
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    h, w = raw.shape[:2]
    results: list[dict[str, Any]] = []
    for contour in contours:
        x, y, cw, ch = cv2.boundingRect(contour)
        area = cw * ch
        if area < max(250, int(w * h * 0.0003)) or cw < 30 or ch < 8:
            continue
        pad_x = max(8, int(cw * 0.18))
        pad_y = max(5, int(ch * 0.55))
        x1, y1 = max(0, x - pad_x), max(0, y - pad_y)
        x2, y2 = min(w, x + cw + pad_x), min(h, y + ch + pad_y)
        roi = raw[y1:y2, x1:x2]
        gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
        gray = cv2.resize(gray, None, fx=4, fy=4, interpolation=cv2.INTER_CUBIC)
        gray = cv2.GaussianBlur(gray, (3, 3), 0)
        _, binary = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        text_candidates = []
        for image in (gray, binary):
            for psm in (6, 7, 11, 13):
                text_candidates.append(_tesseract_text(Image.fromarray(image), "eng", psm, "-c tessedit_char_whitelist=0123456789"))
        values: list[int] = []
        for text in text_candidates:
            for match in re.findall(r"\d{2,7}", text):
                try:
                    value = int(match)
                except ValueError:
                    continue
                if 20 <= value <= 1_000_000:
                    values.append(value)
        if not values:
            continue
        # Prefer the longest and most frequently observed value.
        counts: dict[int, int] = {}
        for value in values:
            counts[value] = counts.get(value, 0) + 1
        best = sorted(counts, key=lambda n: (len(str(n)), counts[n], n), reverse=True)[0]
        results.append({
            "value": best,
            "confidence": min(0.98, 0.72 + 0.04 * counts[best]),
            "source": "yellow highlighted region",
            "bbox": [x1, y1, x2, y2],
        })
    # Deduplicate values while preserving the strongest evidence.
    dedup: dict[int, dict[str, Any]] = {}
    for item in results:
        value = int(item["value"])
        if value not in dedup or item["confidence"] > dedup[value]["confidence"]:
            dedup[value] = item
    return sorted(dedup.values(), key=lambda x: x["confidence"], reverse=True)


def run_ocr(path: str | Path, language: str = "eng+hin") -> dict[str, Any]:
    configure_tesseract()
    image = preprocess_image(path)
    # One layout pass keeps intake fast. Identity evidence is cropped separately for
    # local vision or human review rather than repeatedly applying Tesseract to cursive text.
    config = "--oem 3 --psm 6"
    try:
        data = pytesseract.image_to_data(image, lang=language, config=config, output_type=pytesseract.Output.DICT)
        text = pytesseract.image_to_string(image, lang=language, config=config)
    except pytesseract.TesseractError:
        data = pytesseract.image_to_data(image, lang="eng", config=config, output_type=pytesseract.Output.DICT)
        text = pytesseract.image_to_string(image, lang="eng", config=config)

    data = data or {}
    confidences = []
    boxes = []
    for i, raw_conf in enumerate(data.get("conf", [])):
        try:
            conf = float(raw_conf)
        except (TypeError, ValueError):
            continue
        token = (data.get("text") or [""])[i].strip()
        if conf >= 0 and token:
            confidences.append(conf)
            boxes.append({
                "text": token,
                "confidence": conf,
                "left": int(data["left"][i]),
                "top": int(data["top"][i]),
                "width": int(data["width"][i]),
                "height": int(data["height"][i]),
            })
    avg = sum(confidences) / len(confidences) if confidences else 0.0
    identity_candidates, identity_crops = _ocr_identity_regions(path, text)
    highlighted_numbers = _yellow_highlighted_numbers(path)
    return {
        "text": text.strip(),
        "confidence": round(avg / 100.0, 3),
        "boxes": boxes,
        "identity_candidates": identity_candidates,
        "identity_crops": identity_crops,
        "highlighted_numbers": highlighted_numbers,
    }
