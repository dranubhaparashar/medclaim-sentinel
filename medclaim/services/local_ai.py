from __future__ import annotations

import base64
import json
import os
import re
import time
import urllib.error
import urllib.request
from io import BytesIO
from pathlib import Path
from typing import Any

from PIL import Image, ImageOps

CONFIG_PATH = Path("local_ai_config.json")
DEFAULT_CONFIG: dict[str, Any] = {
    "enabled": True,
    "ollama_url": "http://127.0.0.1:11434",
    "vision_model": "qwen2.5vl:7b",
    "clinical_model": "qwen2.5vl:7b",
    "timeout_seconds": 150,
    "max_image_dimension": 1280,
    "minimum_evidence_confidence": 0.55,
    "num_ctx": 8192,
    "num_predict": 1800,
    "clinical_attach_images": False,
    "skip_existing_document_ai": True,
}

CLINICAL_STATUSES = {
    "MATCHED",
    "PARTIALLY_MATCHED",
    "CONDITIONAL_MATCH",
    "MISMATCH",
    "INSUFFICIENT_EVIDENCE",
    "MEDICAL_OFFICER_REVIEW",
}

DOCUMENT_SCHEMA: dict[str, Any] = {
    "type": "object",
    "properties": {
        "document_type": {"type": "string"},
        "patient_names": {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {
                    "value": {"type": "string"},
                    "confidence": {"type": "number"},
                    "source_text": {"type": "string"},
                },
                "required": ["value", "confidence", "source_text"],
            },
        },
        "providers": {"type": "array", "items": {"type": "string"}},
        "diagnoses": {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {
                    "value": {"type": "string"},
                    "confidence": {"type": "number"},
                    "source_text": {"type": "string"},
                },
                "required": ["value", "confidence", "source_text"],
            },
        },
        "prescribed_medicines": {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {
                    "name": {"type": "string"},
                    "strength": {"type": "string"},
                    "dose": {"type": "string"},
                    "frequency": {"type": "string"},
                    "duration": {"type": "string"},
                    "confidence": {"type": "number"},
                    "source_text": {"type": "string"},
                },
                "required": ["name", "strength", "dose", "frequency", "duration", "confidence", "source_text"],
            },
        },
        "billed_medicines": {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {
                    "name": {"type": "string"},
                    "strength": {"type": "string"},
                    "quantity": {"type": "string"},
                    "amount": {"type": "number"},
                    "confidence": {"type": "number"},
                    "source_text": {"type": "string"},
                },
                "required": ["name", "strength", "quantity", "amount", "confidence", "source_text"],
            },
        },
        "tests": {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {
                    "name": {"type": "string"},
                    "date": {"type": "string"},
                    "amount": {"type": "number"},
                    "confidence": {"type": "number"},
                    "source_text": {"type": "string"},
                },
                "required": ["name", "date", "amount", "confidence", "source_text"],
            },
        },
        "bills": {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {
                    "bill_no": {"type": "string"},
                    "date": {"type": "string"},
                    "amount": {"type": "number"},
                    "dealer": {"type": "string"},
                    "confidence": {"type": "number"},
                    "source_text": {"type": "string"},
                },
                "required": ["bill_no", "date", "amount", "dealer", "confidence", "source_text"],
            },
        },
        "totals": {
            "type": "object",
            "properties": {
                "medicine_total": {"type": "number"},
                "test_total": {"type": "number"},
                "grand_total": {"type": "number"},
            },
            "required": ["medicine_total", "test_total", "grand_total"],
        },
        "dates": {"type": "array", "items": {"type": "string"}},
        "application_numbers": {"type": "array", "items": {"type": "string"}},
        "policy_numbers": {"type": "array", "items": {"type": "string"}},
        "itemized_medicine_names_present": {"type": "boolean"},
        "evidence_quality": {"type": "string"},
        "warnings": {"type": "array", "items": {"type": "string"}},
    },
    "required": [
        "document_type", "patient_names", "providers", "diagnoses",
        "prescribed_medicines", "billed_medicines", "tests", "bills",
        "totals", "dates", "application_numbers", "policy_numbers",
        "itemized_medicine_names_present", "evidence_quality", "warnings",
    ],
}

CLINICAL_SCHEMA: dict[str, Any] = {
    "type": "object",
    "properties": {
        "overall_status": {"type": "string", "enum": sorted(CLINICAL_STATUSES)},
        "overall_explanation": {"type": "string"},
        "diagnoses_used": {"type": "array", "items": {"type": "string"}},
        "medicine_matches": {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {
                    "medicine": {"type": "string"},
                    "source": {"type": "string"},
                    "normalized_name": {"type": "string"},
                    "status": {"type": "string", "enum": sorted(CLINICAL_STATUSES)},
                    "matched_diagnosis": {"type": "string"},
                    "rationale": {"type": "string"},
                    "required_evidence": {"type": "string"},
                    "confidence": {"type": "number"},
                },
                "required": [
                    "medicine", "source", "normalized_name", "status",
                    "matched_diagnosis", "rationale", "required_evidence", "confidence",
                ],
            },
        },
        "investigation_matches": {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {
                    "investigation": {"type": "string"},
                    "status": {"type": "string", "enum": sorted(CLINICAL_STATUSES)},
                    "matched_diagnosis": {"type": "string"},
                    "rationale": {"type": "string"},
                    "required_evidence": {"type": "string"},
                    "confidence": {"type": "number"},
                },
                "required": [
                    "investigation", "status", "matched_diagnosis", "rationale",
                    "required_evidence", "confidence",
                ],
            },
        },
        "prescription_bill_matches": {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {
                    "prescribed": {"type": "string"},
                    "billed": {"type": "string"},
                    "status": {"type": "string", "enum": sorted(CLINICAL_STATUSES)},
                    "rationale": {"type": "string"},
                    "confidence": {"type": "number"},
                },
                "required": ["prescribed", "billed", "status", "rationale", "confidence"],
            },
        },
        "missing_evidence": {"type": "array", "items": {"type": "string"}},
        "model_warning": {"type": "string"},
    },
    "required": [
        "overall_status", "overall_explanation", "diagnoses_used",
        "medicine_matches", "investigation_matches", "prescription_bill_matches",
        "missing_evidence", "model_warning",
    ],
}


def load_config() -> dict[str, Any]:
    config = dict(DEFAULT_CONFIG)
    if CONFIG_PATH.exists():
        try:
            loaded = json.loads(CONFIG_PATH.read_text(encoding="utf-8"))
            if isinstance(loaded, dict):
                config.update(loaded)
        except (OSError, json.JSONDecodeError):
            pass
    env_map = {
        "enabled": os.getenv("ENABLE_LOCAL_AI"),
        "ollama_url": os.getenv("OLLAMA_URL"),
        "vision_model": os.getenv("OLLAMA_VISION_MODEL"),
        "clinical_model": os.getenv("OLLAMA_CLINICAL_MODEL"),
    }
    for key, value in env_map.items():
        if value is None or value == "":
            continue
        if key == "enabled":
            config[key] = value.strip().lower() not in {"0", "false", "no", "off"}
        else:
            config[key] = value.strip()
    return config


def save_config(config: dict[str, Any]) -> None:
    merged = dict(DEFAULT_CONFIG)
    merged.update(config)
    CONFIG_PATH.write_text(json.dumps(merged, indent=2, ensure_ascii=False), encoding="utf-8")


def _request_json(url: str, *, method: str = "GET", payload: dict[str, Any] | None = None, timeout: int = 15) -> dict[str, Any]:
    data = None
    headers = {"Content-Type": "application/json"}
    if payload is not None:
        data = json.dumps(payload, ensure_ascii=False).encode("utf-8")
    request = urllib.request.Request(url, data=data, headers=headers, method=method)
    with urllib.request.urlopen(request, timeout=timeout) as response:  # noqa: S310 - local configurable endpoint
        raw = response.read().decode("utf-8", errors="replace")
    parsed = json.loads(raw)
    if not isinstance(parsed, dict):
        raise ValueError("Ollama returned a non-object response")
    return parsed


def get_ollama_status() -> dict[str, Any]:
    config = load_config()
    base = str(config["ollama_url"]).rstrip("/")
    try:
        payload = _request_json(f"{base}/api/tags", timeout=4)
        models = []
        for item in payload.get("models", []) or []:
            if isinstance(item, dict):
                name = item.get("name") or item.get("model")
                if name:
                    models.append(str(name))
        requested = str(config.get("vision_model") or "")
        installed = any(name == requested or name.startswith(requested + ":") for name in models)
        return {
            "available": True,
            "enabled": bool(config.get("enabled")),
            "url": base,
            "models": models,
            "vision_model": requested,
            "clinical_model": str(config.get("clinical_model") or requested),
            "requested_model_installed": installed,
            "error": "",
        }
    except Exception as exc:  # network errors should not stop claim intake
        return {
            "available": False,
            "enabled": bool(config.get("enabled")),
            "url": base,
            "models": [],
            "vision_model": str(config.get("vision_model") or ""),
            "clinical_model": str(config.get("clinical_model") or ""),
            "requested_model_installed": False,
            "error": str(exc),
        }


def _image_to_base64(path: Path, max_dimension: int) -> str:
    with Image.open(path) as image:
        image = ImageOps.exif_transpose(image).convert("RGB")
        image.thumbnail((max_dimension, max_dimension), Image.Resampling.LANCZOS)
        buffer = BytesIO()
        image.save(buffer, format="JPEG", quality=92, optimize=True)
    return base64.b64encode(buffer.getvalue()).decode("ascii")


def _parse_model_json(content: Any) -> dict[str, Any]:
    if isinstance(content, dict):
        return content
    text = str(content or "").strip()
    text = re.sub(r"^```(?:json)?\s*", "", text, flags=re.I)
    text = re.sub(r"\s*```$", "", text)
    parsed = json.loads(text)
    if not isinstance(parsed, dict):
        raise ValueError("Model response was not a JSON object")
    return parsed


def _chat(
    *,
    model: str,
    prompt: str,
    schema: dict[str, Any],
    image_paths: list[Path] | None = None,
) -> tuple[dict[str, Any], dict[str, Any]]:
    config = load_config()
    base = str(config["ollama_url"]).rstrip("/")
    images = []
    for path in image_paths or []:
        if path.exists():
            images.append(_image_to_base64(path, int(config.get("max_image_dimension") or 1800)))
    message: dict[str, Any] = {"role": "user", "content": prompt}
    if images:
        message["images"] = images
    payload = {
        "model": model,
        "messages": [message],
        "format": schema,
        "stream": False,
        "options": {
            "temperature": 0,
            "seed": 42,
            "num_ctx": int(config.get("num_ctx") or 8192),
            "num_predict": int(config.get("num_predict") or 1800),
        },
        "keep_alive": "10m",
    }
    started = time.monotonic()
    response = _request_json(
        f"{base}/api/chat",
        method="POST",
        payload=payload,
        timeout=int(config.get("timeout_seconds") or 420),
    )
    elapsed = time.monotonic() - started
    content = (response.get("message") or {}).get("content")
    parsed = _parse_model_json(content)
    metrics = {
        "model": response.get("model") or model,
        "elapsed_seconds": round(elapsed, 2),
        "prompt_eval_count": response.get("prompt_eval_count"),
        "eval_count": response.get("eval_count"),
        "total_duration_ns": response.get("total_duration"),
    }
    return parsed, metrics


def _safe_number(value: Any) -> float:
    try:
        number = float(value or 0)
    except (TypeError, ValueError):
        return 0.0
    return number if number >= 0 else 0.0


def _clean_list(values: Any) -> list[Any]:
    return values if isinstance(values, list) else []


def normalize_document_result(result: dict[str, Any]) -> dict[str, Any]:
    normalized = dict(result)
    for key in ("patient_names", "providers", "diagnoses", "prescribed_medicines", "billed_medicines", "tests", "bills", "dates", "application_numbers", "policy_numbers", "warnings"):
        normalized[key] = _clean_list(result.get(key))
    totals = result.get("totals") if isinstance(result.get("totals"), dict) else {}
    normalized["totals"] = {
        "medicine_total": _safe_number(totals.get("medicine_total")),
        "test_total": _safe_number(totals.get("test_total")),
        "grand_total": _safe_number(totals.get("grand_total")),
    }
    normalized["itemized_medicine_names_present"] = bool(result.get("itemized_medicine_names_present"))
    normalized["evidence_quality"] = str(result.get("evidence_quality") or "unknown")
    normalized["document_type"] = str(result.get("document_type") or "supporting_document")
    return normalized


def extract_document_with_local_ai(path: Path, *, filename: str, ocr_text: str = "") -> dict[str, Any]:
    config = load_config()
    if not config.get("enabled"):
        return {"ok": False, "error": "Local AI is disabled", "data": {}, "metrics": {}}
    status = get_ollama_status()
    if not status["available"]:
        return {"ok": False, "error": f"Ollama is not reachable at {status['url']}", "data": {}, "metrics": {}}
    model = str(config.get("vision_model") or "qwen2.5vl:7b")
    installed = status.get("models", [])
    if installed and not any(x == model or x.startswith(model + ":") for x in installed):
        return {"ok": False, "error": f"Model '{model}' is not installed. Run: ollama pull {model}", "data": {}, "metrics": {}}

    prompt = f"""
You are a medical-claim document extraction engine operating locally. Read the attached image carefully.
Return only JSON matching the supplied schema.

Strict evidence rules:
1. Extract only text or table values visibly supported by this image. Never invent a medicine, diagnosis, patient name, strength, dose, quantity, bill number, date or amount.
2. Distinguish medicines from laboratory investigations. HbA1c, TSH, T3, T4, vitamin B12 serum test, vitamin D serum test, CBC, FBS, PPBS and ultrasound scans are investigations unless the document explicitly shows a drug/product formulation.
3. A pharmacy summary containing only bill number, date and amount is NOT an itemized medicine invoice. In that case set itemized_medicine_names_present=false and billed_medicines=[] even though medicine bill totals exist.
4. A prescription medicine requires a visible medicine name. Preserve uncertain handwriting in source_text, lower its confidence, and add a warning.
5. Patient names must come from a labelled patient/beneficiary field or clear identity line; do not treat a signature alone as a verified patient name.
6. Amount 0 means absent/not found, not a guessed value.
7. Dates should be copied as visible using DD-MM-YYYY when safely normalizable; otherwise preserve the visible date.
8. source_text must contain the short visible phrase/row that supports each extracted item.

File name: {filename}
Existing conventional OCR text may help, but the image is authoritative:
--- OCR START ---
{ocr_text[:12000]}
--- OCR END ---
""".strip()
    try:
        raw, metrics = _chat(model=model, prompt=prompt, schema=DOCUMENT_SCHEMA, image_paths=[path])
        return {"ok": True, "error": "", "data": normalize_document_result(raw), "metrics": metrics}
    except (urllib.error.URLError, TimeoutError, json.JSONDecodeError, ValueError, OSError) as exc:
        return {"ok": False, "error": str(exc), "data": {}, "metrics": {}}


def _medicine_display(item: Any) -> str:
    if isinstance(item, str):
        return item.strip()
    if not isinstance(item, dict):
        return ""
    parts = [str(item.get("name") or "").strip(), str(item.get("strength") or "").strip()]
    return " ".join(x for x in parts if x).strip()


def merge_local_ai_into_extracted(extracted: dict[str, Any], local_result: dict[str, Any]) -> dict[str, Any]:
    """Merge only evidence-backed local-AI fields into conventional extraction.

    The raw model output remains under ``local_ai`` for auditability. Existing reviewed
    sample values and deterministic extraction are not deleted.
    """
    merged = dict(extracted)
    if not local_result.get("ok"):
        merged["local_ai"] = {
            "ok": False,
            "error": local_result.get("error") or "Local AI failed",
            "metrics": local_result.get("metrics") or {},
        }
        return merged

    data = normalize_document_result(local_result.get("data") or {})
    config = load_config()
    min_conf = float(config.get("minimum_evidence_confidence") or 0.55)
    merged["local_ai"] = {
        "ok": True,
        "data": data,
        "metrics": local_result.get("metrics") or {},
        "processed_at_model": (local_result.get("metrics") or {}).get("model"),
    }

    doc_type = data.get("document_type")
    if doc_type and doc_type != "supporting_document":
        merged["document_type"] = doc_type

    identity_candidates = list(merged.get("identity_candidates") or [])
    for candidate in data.get("patient_names", []):
        if not isinstance(candidate, dict):
            continue
        value = str(candidate.get("value") or "").strip()
        confidence = _safe_number(candidate.get("confidence"))
        if value and confidence >= min_conf:
            identity_candidates.append({
                "value": value,
                "confidence": min(confidence, 1.0),
                "source": f"local AI: {candidate.get('source_text') or 'visible identity field'}",
            })
    if identity_candidates:
        merged["identity_candidates"] = identity_candidates

    providers = [str(x).strip() for x in data.get("providers", []) if str(x).strip()]
    if providers and not merged.get("provider"):
        merged["provider"] = " / ".join(dict.fromkeys(providers))

    diagnoses = [x for x in data.get("diagnoses", []) if isinstance(x, dict) and str(x.get("value") or "").strip()]
    if diagnoses:
        best = max(diagnoses, key=lambda x: _safe_number(x.get("confidence")))
        if _safe_number(best.get("confidence")) >= min_conf:
            merged["diagnosis"] = str(best.get("value")).strip()
        merged["diagnosis_candidates"] = diagnoses

    prescribed = []
    for item in data.get("prescribed_medicines", []):
        if not isinstance(item, dict) or _safe_number(item.get("confidence")) < min_conf:
            continue
        display = _medicine_display(item)
        if display:
            prescribed.append(display)
    if prescribed:
        merged["prescribed_medicines"] = list(dict.fromkeys((merged.get("prescribed_medicines") or []) + prescribed))

    billed = []
    if data.get("itemized_medicine_names_present"):
        for item in data.get("billed_medicines", []):
            if not isinstance(item, dict) or _safe_number(item.get("confidence")) < min_conf:
                continue
            display = _medicine_display(item)
            if display:
                billed.append(display)
    if billed:
        merged["billed_medicines"] = list(dict.fromkeys((merged.get("billed_medicines") or []) + billed))
        merged["medicine_items"] = data.get("billed_medicines", [])

    local_tests = []
    for item in data.get("tests", []):
        if not isinstance(item, dict) or not str(item.get("name") or "").strip():
            continue
        if _safe_number(item.get("confidence")) < min_conf:
            continue
        local_tests.append({
            "name": str(item.get("name")).strip(),
            "date": str(item.get("date") or "").strip(),
            "amount": _safe_number(item.get("amount")),
            "confidence": min(_safe_number(item.get("confidence")), 1.0),
            "source_text": str(item.get("source_text") or "").strip(),
        })
    if local_tests:
        existing = list(merged.get("tests") or [])
        seen = {(str(x.get("name", "")).lower(), str(x.get("date", "")), str(x.get("amount", ""))) for x in existing if isinstance(x, dict)}
        for item in local_tests:
            key = (item["name"].lower(), item["date"], str(item["amount"]))
            if key not in seen:
                existing.append(item)
                seen.add(key)
        merged["tests"] = existing

    bills = []
    for item in data.get("bills", []):
        if not isinstance(item, dict) or _safe_number(item.get("confidence")) < min_conf:
            continue
        bill_no = str(item.get("bill_no") or "").strip()
        date = str(item.get("date") or "").strip()
        amount = _safe_number(item.get("amount"))
        if bill_no or date or amount:
            bills.append({"invoice_no": bill_no, "date": date, "amount": amount, "dealer": str(item.get("dealer") or "").strip()})
    if bills:
        if data.get("document_type") == "medicine_bill_summary":
            merged["medicine_bills"] = bills
        elif data.get("document_type") == "laboratory_bill_summary":
            merged["test_bill_rows"] = bills
        else:
            merged["bill_rows"] = bills
        merged["invoice_numbers"] = sorted({x["invoice_no"] for x in bills if x["invoice_no"]} | set(merged.get("invoice_numbers") or []))

    dates = [str(x).strip() for x in data.get("dates", []) if str(x).strip()]
    if dates:
        merged["dates"] = sorted(set(merged.get("dates") or []) | set(dates))
    applications = [str(x).strip() for x in data.get("application_numbers", []) if str(x).strip()]
    policies = [str(x).strip() for x in data.get("policy_numbers", []) if str(x).strip()]
    if applications:
        merged["application_numbers"] = list(dict.fromkeys(applications))
    if policies:
        merged["policy_numbers"] = list(dict.fromkeys(policies))

    totals = data.get("totals") or {}
    if _safe_number(totals.get("medicine_total")) > 0:
        merged["medicine_total"] = _safe_number(totals.get("medicine_total"))
    if _safe_number(totals.get("test_total")) > 0:
        merged["test_total"] = _safe_number(totals.get("test_total"))
    if _safe_number(totals.get("grand_total")) > 0:
        merged["grand_total"] = _safe_number(totals.get("grand_total"))

    merged["itemized_medicine_names_present"] = bool(data.get("itemized_medicine_names_present"))
    merged["local_ai_warnings"] = data.get("warnings") or []
    return merged


def build_claim_evidence(claim: dict[str, Any], documents: list[dict[str, Any]]) -> dict[str, Any]:
    diagnoses: list[str] = []
    prescribed: list[str] = []
    billed: list[str] = []
    tests: list[str] = []
    evidence_docs: list[dict[str, Any]] = []
    itemized_present = False
    for doc in documents:
        ext = doc.get("extracted") or {}
        if ext.get("diagnosis"):
            diagnoses.append(str(ext["diagnosis"]))
        prescribed.extend(str(x) for x in ext.get("prescribed_medicines", []) if str(x).strip())
        billed.extend(str(x) for x in ext.get("billed_medicines", []) if str(x).strip())
        itemized_present = itemized_present or bool(ext.get("itemized_medicine_names_present"))
        for item in ext.get("tests", []) or []:
            if isinstance(item, dict) and item.get("name"):
                tests.append(str(item["name"]))
        evidence_docs.append({
            "filename": doc.get("filename"),
            "document_type": doc.get("doc_type"),
            "diagnosis": ext.get("diagnosis"),
            "prescribed_medicines": ext.get("prescribed_medicines") or [],
            "billed_medicines": ext.get("billed_medicines") or [],
            "tests": ext.get("tests") or [],
            "itemized_medicine_names_present": bool(ext.get("itemized_medicine_names_present")),
            "local_ai_warnings": ext.get("local_ai_warnings") or [],
        })
    if claim.get("diagnosis"):
        diagnoses.insert(0, str(claim["diagnosis"]))
    return {
        "claim_no": claim.get("claim_no"),
        "patient_name": claim.get("patient_name"),
        "diagnoses": list(dict.fromkeys(x.strip() for x in diagnoses if x.strip())),
        "prescribed_medicines": list(dict.fromkeys(x.strip() for x in prescribed if x.strip())),
        "billed_medicines": list(dict.fromkeys(x.strip() for x in billed if x.strip())),
        "investigations": list(dict.fromkeys(x.strip() for x in tests if x.strip())),
        "itemized_medicine_names_present": itemized_present,
        "documents": evidence_docs,
    }



def _postprocess_local_clinical_result(raw: dict[str, Any], evidence: dict[str, Any]) -> dict[str, Any]:
    """Apply non-negotiable evidence gates after the local model responds.

    The local model may explain clinical relationships, but it cannot convert a
    document-presence signal into a medicine match or treat pregnancy alone as proof
    of diabetes, thyroid disease, or a vitamin deficiency.
    """
    data = dict(raw or {})
    diagnoses = " ".join(str(x) for x in evidence.get("diagnoses") or []).lower()
    has_pregnancy = any(x in diagnoses for x in ("pregnan", "antenatal", "prenatal", "gestation"))
    has_diabetes = any(x in diagnoses for x in ("diabetes", "gdm", "hypergly"))
    has_thyroid = any(x in diagnoses for x in ("hypothy", "raised tsh", "thyroid deficiency"))
    has_b12 = "b12 deficiency" in diagnoses or "vitamin b12 deficiency" in diagnoses
    has_vitd = "vitamin d deficiency" in diagnoses or "low vitamin d" in diagnoses

    prescribed = [str(x).strip() for x in evidence.get("prescribed_medicines") or [] if str(x).strip()]
    billed = [str(x).strip() for x in evidence.get("billed_medicines") or [] if str(x).strip()]
    itemized = bool(evidence.get("itemized_medicine_names_present")) and bool(billed)
    missing = list(dict.fromkeys(str(x).strip() for x in data.get("missing_evidence") or [] if str(x).strip()))

    if not prescribed:
        data["medicine_matches"] = []
        missing.append("Readable prescription with medicine name, strength, dose and duration")
    if not itemized:
        data["prescription_bill_matches"] = []
        missing.append("Itemized pharmacy invoice with medicine name, strength, quantity and amount")

    conservative_investigations = []
    for item in data.get("investigation_matches") or []:
        if not isinstance(item, dict):
            continue
        row = dict(item)
        name = str(row.get("investigation") or "").lower()
        status = str(row.get("status") or "MEDICAL_OFFICER_REVIEW")
        # Pregnancy alone provides context, but not a diagnosis of diabetes,
        # thyroid disease, B12 deficiency or vitamin-D deficiency.
        conditional = False
        if has_pregnancy and not has_diabetes and any(k in name for k in ("fbs", "ppbs", "glucose", "hba1c", "a1c")):
            conditional = True
        if has_pregnancy and not has_thyroid and any(k in name for k in ("tsh", "t3", "t4", "thyroid", "thyroxine")):
            conditional = True
        if has_pregnancy and not has_b12 and "b12" in name:
            conditional = True
        if has_pregnancy and not has_vitd and ("vitamin d" in name or "25-oh" in name or "25 oh" in name):
            conditional = True
        if conditional and status == "MATCHED":
            row["status"] = "CONDITIONAL_MATCH"
            row["rationale"] = (str(row.get("rationale") or "") + " Pregnancy alone does not prove the specific comorbidity or deficiency; supporting clinical context is required.").strip()
            row["required_evidence"] = row.get("required_evidence") or "Specific diagnosis, abnormal result, risk factor or prescriber note"
            row["confidence"] = min(_safe_number(row.get("confidence")) or 0.75, 0.85)
        conservative_investigations.append(row)
    data["investigation_matches"] = conservative_investigations

    if not prescribed or not itemized:
        if data.get("overall_status") == "MATCHED":
            data["overall_status"] = "INSUFFICIENT_EVIDENCE"
        data["overall_explanation"] = (
            "Medicine-level matching is incomplete because a readable prescribed-medicine list and/or itemized billed-medicine list is missing. "
            "Investigation relationships may still be reviewed separately."
        )

    data["missing_evidence"] = list(dict.fromkeys(missing))
    warning = str(data.get("model_warning") or "").strip()
    safety = "Local-model output is evidence-gated and cannot approve or reject the claim."
    data["model_warning"] = (warning + " " + safety).strip()
    return data

def run_local_clinical_match(claim: dict[str, Any], documents: list[dict[str, Any]], rules: dict[str, Any]) -> dict[str, Any]:
    config = load_config()
    if not config.get("enabled"):
        return {"ok": False, "error": "Local AI is disabled", "data": {}, "metrics": {}}
    status = get_ollama_status()
    if not status["available"]:
        return {"ok": False, "error": f"Ollama is not reachable at {status['url']}", "data": {}, "metrics": {}}
    model = str(config.get("clinical_model") or config.get("vision_model") or "qwen2.5vl:7b")
    installed = status.get("models", [])
    if installed and not any(x == model or x.startswith(model + ":") for x in installed):
        return {"ok": False, "error": f"Model '{model}' is not installed. Run: ollama pull {model}", "data": {}, "metrics": {}}

    evidence = build_claim_evidence(claim, documents)
    compact_rules = {
        "diagnosis_concepts": rules.get("diagnosis_concepts", []),
        "medicines": rules.get("medicines", []),
        "investigations": rules.get("investigations", []),
        "disclaimer": rules.get("disclaimer", ""),
    }
    attach_images = bool(config.get("clinical_attach_images", False))
    image_paths = [Path(str(d.get("stored_path"))) for d in documents if d.get("stored_path")] if attach_images else []
    evidence_source_note = (
        "Review the attached claim images together with the structured evidence."
        if attach_images
        else "Review the structured evidence extracted from the claim images. Do not request or assume unseen image content."
    )
    prompt = f"""
You are a local clinical-claim consistency reviewer. {evidence_source_note}
Return only JSON matching the supplied schema.

Your task has three independent dimensions:
A. diagnosis/treatment context ↔ prescribed medicine;
B. prescribed medicine ↔ itemized billed medicine;
C. diagnosis/treatment context ↔ investigations/tests.

Mandatory safety and evidence rules:
1. Never infer medicine names from a bill number, amount, pharmacy name, test name or signature.
2. A medicine-bill summary with only bill/date/amount cannot support medicine-level matching. Mark medicine matching INSUFFICIENT_EVIDENCE.
3. Do not call a laboratory investigation a medicine. TSH/T3/T4/thyroxine assay, HbA1c, CBC, vitamin B12 serum and vitamin D serum are tests unless an explicit product/medicine is visible.
4. Use MATCHED only when the medicine/test is clinically compatible with a diagnosis actually evidenced in the bundle.
5. Use CONDITIONAL_MATCH when it can be appropriate only if a missing comorbidity, lab abnormality, pregnancy stage or physician indication is confirmed.
6. Use MISMATCH only for a clear contradiction; otherwise use MEDICAL_OFFICER_REVIEW or INSUFFICIENT_EVIDENCE.
7. Do not approve or reject a claim. Explain the evidence gap and request medical-officer review where needed.
8. Preserve uncertainty. Confidence must reflect image legibility and evidence quality, not confidence in general medical knowledge.
9. Brand/generic normalization is allowed only when you are confident (example: Thyronorm → levothyroxine; Folvite → folic acid).
10. If the attached images disagree with extracted text, prefer clearly visible image evidence and mention the conflict.
11. Pregnancy-specific ultrasound/fetal-screening tests may be MATCHED to antenatal care. CBC and urinalysis may be MATCHED when clearly part of antenatal care.
12. FBS, PPBS, HbA1c, thyroid-profile, vitamin B12 and vitamin D tests must be CONDITIONAL_MATCH when pregnancy is the only documented context; MATCHED requires the corresponding diabetes, thyroid or deficiency evidence.
13. A prescription-type document being present is document completeness only; it is never by itself a clinical medicine match.
14. Every confidence value must be a number between 0 and 1; never return null.

Structured claim evidence:
{json.dumps(evidence, ensure_ascii=False, indent=2)[:30000]}

Local approved rule pack (use as grounding; do not fabricate additional policy rules):
{json.dumps(compact_rules, ensure_ascii=False, indent=2)[:30000]}
""".strip()
    try:
        raw, metrics = _chat(model=model, prompt=prompt, schema=CLINICAL_SCHEMA, image_paths=image_paths[:8])
        metrics["images_attached"] = len(image_paths[:8])
        raw = _postprocess_local_clinical_result(raw, evidence)
        if raw.get("overall_status") not in CLINICAL_STATUSES:
            raw["overall_status"] = "MEDICAL_OFFICER_REVIEW"
        return {"ok": True, "error": "", "data": raw, "metrics": metrics, "evidence": evidence}
    except (urllib.error.URLError, TimeoutError, json.JSONDecodeError, ValueError, OSError) as exc:
        return {"ok": False, "error": str(exc), "data": {}, "metrics": {}, "evidence": evidence}



def _local_result_confidence(data: dict[str, Any]) -> float:
    values: list[float] = []
    for key in ("medicine_matches", "investigation_matches", "prescription_bill_matches"):
        for item in data.get(key) or []:
            if isinstance(item, dict):
                value = _safe_number(item.get("confidence"))
                if 0 < value <= 1:
                    values.append(value)
    if not values:
        return 0.60
    return round(sum(values) / len(values), 3)

def local_clinical_result_to_checks(result: dict[str, Any]) -> list[dict[str, Any]]:
    if not result.get("ok"):
        return [{
            "check_code": "LOCAL_AI_UNAVAILABLE",
            "severity": "low",
            "result": "information",
            "explanation": result.get("error") or "Local AI clinical matching did not run.",
            "evidence": {
                "clinical_status": "INFORMATION",
                "dimension": "Local AI runtime",
                "item": "Ollama clinical model",
                "confidence": 1.0,
                "required_evidence": "Start Ollama and install the configured vision model.",
                "raw_evidence": result.get("metrics") or {},
            },
        }]

    data = result.get("data") or {}
    metrics = result.get("metrics") or {}
    status = str(data.get("overall_status") or "MEDICAL_OFFICER_REVIEW")
    severity = "high" if status == "MISMATCH" else "medium" if status not in {"MATCHED"} else "low"
    checks: list[dict[str, Any]] = [{
        "check_code": "LOCAL_AI_CLINICAL_SUMMARY",
        "severity": severity,
        "result": "consistent" if status == "MATCHED" else "review",
        "explanation": str(data.get("overall_explanation") or "Local model completed clinical consistency review."),
        "evidence": {
            "clinical_status": status,
            "dimension": "Local VLM clinical assessment",
            "item": "Diagnosis–medicine–bill–investigation consistency",
            "confidence": _local_result_confidence(data),
            "required_evidence": "; ".join(data.get("missing_evidence") or []),
            "raw_evidence": {"diagnoses_used": data.get("diagnoses_used") or [], "model_warning": data.get("model_warning") or "", "metrics": metrics},
            "engine": "local_ai_cross_check",
        },
    }]

    for index, item in enumerate(data.get("medicine_matches") or [], start=1):
        if not isinstance(item, dict):
            continue
        item_status = str(item.get("status") or "MEDICAL_OFFICER_REVIEW")
        checks.append({
            "check_code": f"LOCAL_AI_MEDICINE_MATCH_{index}",
            "severity": "high" if item_status == "MISMATCH" else "medium" if item_status != "MATCHED" else "low",
            "result": "consistent" if item_status == "MATCHED" else "review",
            "explanation": str(item.get("rationale") or "Local model medicine assessment."),
            "evidence": {
                "clinical_status": item_status,
                "dimension": "Diagnosis ↔ medicine",
                "item": item.get("medicine") or "Unknown medicine",
                "confidence": _safe_number(item.get("confidence")),
                "required_evidence": item.get("required_evidence") or "",
                "raw_evidence": {
                    "source": item.get("source"),
                    "normalized_name": item.get("normalized_name"),
                    "matched_diagnosis": item.get("matched_diagnosis"),
                },
                "engine": "local_ai_cross_check",
            },
        })

    for index, item in enumerate(data.get("investigation_matches") or [], start=1):
        if not isinstance(item, dict):
            continue
        item_status = str(item.get("status") or "MEDICAL_OFFICER_REVIEW")
        checks.append({
            "check_code": f"LOCAL_AI_INVESTIGATION_MATCH_{index}",
            "severity": "high" if item_status == "MISMATCH" else "medium" if item_status != "MATCHED" else "low",
            "result": "consistent" if item_status == "MATCHED" else "review",
            "explanation": str(item.get("rationale") or "Local model investigation assessment."),
            "evidence": {
                "clinical_status": item_status,
                "dimension": "Diagnosis ↔ investigation",
                "item": item.get("investigation") or "Unknown investigation",
                "confidence": _safe_number(item.get("confidence")),
                "required_evidence": item.get("required_evidence") or "",
                "raw_evidence": {"matched_diagnosis": item.get("matched_diagnosis")},
                "engine": "local_ai_cross_check",
            },
        })

    for index, item in enumerate(data.get("prescription_bill_matches") or [], start=1):
        if not isinstance(item, dict):
            continue
        item_status = str(item.get("status") or "MEDICAL_OFFICER_REVIEW")
        checks.append({
            "check_code": f"LOCAL_AI_PRESCRIPTION_BILL_MATCH_{index}",
            "severity": "high" if item_status == "MISMATCH" else "medium" if item_status != "MATCHED" else "low",
            "result": "consistent" if item_status == "MATCHED" else "review",
            "explanation": str(item.get("rationale") or "Local model prescription-to-bill assessment."),
            "evidence": {
                "clinical_status": item_status,
                "dimension": "Prescription ↔ pharmacy bill",
                "item": item.get("billed") or item.get("prescribed") or "Medicine item",
                "confidence": _safe_number(item.get("confidence")),
                "required_evidence": "",
                "raw_evidence": {"prescribed": item.get("prescribed"), "billed": item.get("billed")},
                "engine": "local_ai_cross_check",
            },
        })
    return checks
