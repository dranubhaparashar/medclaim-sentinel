from __future__ import annotations

import json
import shutil
import sqlite3
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable

DB_PATH = Path("storage/medclaim.db")


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def connect() -> sqlite3.Connection:
    DB_PATH.parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(DB_PATH, check_same_thread=False)
    conn.row_factory = sqlite3.Row
    return conn


def init_db() -> None:
    with connect() as conn:
        conn.executescript(
            """
            CREATE TABLE IF NOT EXISTS claims (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                claim_no TEXT UNIQUE NOT NULL,
                application_no TEXT,
                patient_name TEXT NOT NULL,
                policy_no TEXT,
                hospital TEXT,
                diagnosis TEXT,
                treatment_start TEXT,
                treatment_end TEXT,
                claimed_amount REAL DEFAULT 0,
                supported_amount REAL DEFAULT 0,
                recommended_amount REAL DEFAULT 0,
                risk_score INTEGER DEFAULT 0,
                risk_band TEXT DEFAULT 'pending',
                status TEXT DEFAULT 'Received',
                assigned_to TEXT,
                created_at TEXT NOT NULL,
                updated_at TEXT NOT NULL
            );

            CREATE TABLE IF NOT EXISTS documents (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                claim_id INTEGER NOT NULL,
                filename TEXT NOT NULL,
                stored_path TEXT NOT NULL,
                sha256 TEXT,
                phash TEXT,
                doc_type TEXT DEFAULT 'unknown',
                ocr_text TEXT DEFAULT '',
                ocr_confidence REAL DEFAULT 0,
                extracted_json TEXT DEFAULT '{}',
                created_at TEXT NOT NULL,
                FOREIGN KEY(claim_id) REFERENCES claims(id)
            );

            CREATE TABLE IF NOT EXISTS duplicate_matches (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                claim_id INTEGER NOT NULL,
                matched_claim_id INTEGER,
                matched_document_id INTEGER,
                score REAL NOT NULL,
                status TEXT NOT NULL,
                reasons_json TEXT NOT NULL,
                created_at TEXT NOT NULL
            );

            CREATE TABLE IF NOT EXISTS clinical_checks (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                claim_id INTEGER NOT NULL,
                check_code TEXT NOT NULL,
                severity TEXT NOT NULL,
                result TEXT NOT NULL,
                explanation TEXT NOT NULL,
                evidence_json TEXT DEFAULT '[]',
                created_at TEXT NOT NULL
            );

            CREATE TABLE IF NOT EXISTS claim_clinical_inputs (
                claim_id INTEGER PRIMARY KEY,
                diagnosis_text TEXT DEFAULT '',
                prescribed_medicines_json TEXT DEFAULT '[]',
                billed_medicines_json TEXT DEFAULT '[]',
                reviewer_notes TEXT DEFAULT '',
                updated_by TEXT DEFAULT 'claim_reviewer',
                updated_at TEXT NOT NULL,
                FOREIGN KEY(claim_id) REFERENCES claims(id)
            );

            CREATE TABLE IF NOT EXISTS correction_proposals (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                claim_id INTEGER NOT NULL,
                document_id INTEGER,
                field_name TEXT NOT NULL,
                current_value TEXT,
                proposed_value TEXT NOT NULL,
                reason TEXT,
                status TEXT DEFAULT 'Pending',
                requested_by TEXT DEFAULT 'chat_user',
                reviewed_by TEXT,
                created_at TEXT NOT NULL,
                reviewed_at TEXT
            );

            CREATE TABLE IF NOT EXISTS audit_events (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                claim_id INTEGER,
                actor TEXT NOT NULL,
                action TEXT NOT NULL,
                details_json TEXT NOT NULL,
                created_at TEXT NOT NULL
            );

            CREATE TABLE IF NOT EXISTS claim_deletion_log (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                claim_no TEXT NOT NULL,
                patient_name TEXT,
                status TEXT,
                document_count INTEGER DEFAULT 0,
                reason TEXT,
                deleted_by TEXT NOT NULL,
                summary_json TEXT NOT NULL,
                deleted_at TEXT NOT NULL
            );
            """
        )


def audit(claim_id: int | None, actor: str, action: str, details: dict[str, Any]) -> None:
    with connect() as conn:
        conn.execute(
            "INSERT INTO audit_events (claim_id, actor, action, details_json, created_at) VALUES (?, ?, ?, ?, ?)",
            (claim_id, actor, action, json.dumps(details, ensure_ascii=False), utc_now()),
        )


def create_claim(data: dict[str, Any]) -> int:
    now = utc_now()
    with connect() as conn:
        cur = conn.execute(
            """
            INSERT INTO claims (
                claim_no, application_no, patient_name, policy_no, hospital, diagnosis,
                treatment_start, treatment_end, claimed_amount, status, created_at, updated_at
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                data["claim_no"], data.get("application_no"), data["patient_name"],
                data.get("policy_no"), data.get("hospital"), data.get("diagnosis"),
                data.get("treatment_start"), data.get("treatment_end"),
                float(data.get("claimed_amount") or 0), data.get("status", "Received"), now, now,
            ),
        )
        claim_id = int(cur.lastrowid)
    audit(claim_id, "system", "CLAIM_CREATED", data)
    return claim_id


def claim_exists(claim_no: str) -> bool:
    with connect() as conn:
        return conn.execute("SELECT 1 FROM claims WHERE claim_no = ?", (claim_no,)).fetchone() is not None


def get_claim(claim_id: int) -> dict[str, Any] | None:
    with connect() as conn:
        row = conn.execute("SELECT * FROM claims WHERE id = ?", (claim_id,)).fetchone()
    return dict(row) if row else None


def get_claim_by_no(claim_no: str) -> dict[str, Any] | None:
    with connect() as conn:
        row = conn.execute("SELECT * FROM claims WHERE claim_no = ?", (claim_no,)).fetchone()
    return dict(row) if row else None


def list_claims() -> list[dict[str, Any]]:
    with connect() as conn:
        rows = conn.execute("SELECT * FROM claims ORDER BY id DESC").fetchall()
    return [dict(r) for r in rows]


def delete_claim(
    claim_id: int,
    *,
    deleted_by: str = "claim_reviewer",
    reason: str = "Removed by reviewer",
    delete_files: bool = True,
) -> dict[str, Any]:
    """Permanently remove one claim and all claim-owned records.

    A minimal deletion log is retained for governance, while uploaded files and all
    active claim data are removed. Duplicate links from other claims are also deleted
    so they can be recalculated without pointing to a missing claim.
    """
    claim = get_claim(claim_id)
    if not claim:
        raise ValueError("Claim not found")

    documents = list_documents(claim_id)
    with connect() as conn:
        affected_rows = conn.execute(
            "SELECT DISTINCT claim_id FROM duplicate_matches "
            "WHERE matched_claim_id = ? AND claim_id <> ?",
            (claim_id, claim_id),
        ).fetchall()
        affected_claim_ids = [int(row["claim_id"]) for row in affected_rows]

        counts = {
            "documents": int(conn.execute("SELECT COUNT(*) FROM documents WHERE claim_id = ?", (claim_id,)).fetchone()[0]),
            "duplicate_matches": int(conn.execute(
                "SELECT COUNT(*) FROM duplicate_matches WHERE claim_id = ? OR matched_claim_id = ?",
                (claim_id, claim_id),
            ).fetchone()[0]),
            "clinical_checks": int(conn.execute("SELECT COUNT(*) FROM clinical_checks WHERE claim_id = ?", (claim_id,)).fetchone()[0]),
            "corrections": int(conn.execute("SELECT COUNT(*) FROM correction_proposals WHERE claim_id = ?", (claim_id,)).fetchone()[0]),
            "audit_events": int(conn.execute("SELECT COUNT(*) FROM audit_events WHERE claim_id = ?", (claim_id,)).fetchone()[0]),
        }
        summary = {
            "claim_id": claim_id,
            "claim_no": claim.get("claim_no"),
            "status": claim.get("status"),
            "claimed_amount": claim.get("claimed_amount"),
            "supported_amount": claim.get("supported_amount"),
            "record_counts": counts,
            "affected_claim_ids": affected_claim_ids,
        }
        conn.execute(
            """INSERT INTO claim_deletion_log
            (claim_no, patient_name, status, document_count, reason, deleted_by, summary_json, deleted_at)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?)""",
            (
                str(claim.get("claim_no") or ""),
                str(claim.get("patient_name") or ""),
                str(claim.get("status") or ""),
                counts["documents"],
                reason.strip(),
                deleted_by,
                json.dumps(summary, ensure_ascii=False),
                utc_now(),
            ),
        )

        conn.execute("DELETE FROM correction_proposals WHERE claim_id = ?", (claim_id,))
        conn.execute("DELETE FROM claim_clinical_inputs WHERE claim_id = ?", (claim_id,))
        conn.execute("DELETE FROM clinical_checks WHERE claim_id = ?", (claim_id,))
        conn.execute(
            "DELETE FROM duplicate_matches WHERE claim_id = ? OR matched_claim_id = ?",
            (claim_id, claim_id),
        )
        conn.execute("DELETE FROM documents WHERE claim_id = ?", (claim_id,))
        conn.execute("DELETE FROM audit_events WHERE claim_id = ?", (claim_id,))
        conn.execute("DELETE FROM claims WHERE id = ?", (claim_id,))

    removed_files = 0
    file_errors: list[str] = []
    if delete_files:
        storage_root = (Path("storage") / "claims").resolve()
        claim_folder = (storage_root / str(claim_id)).resolve()
        try:
            claim_folder.relative_to(storage_root)
            if claim_folder.exists():
                removed_files = sum(1 for path in claim_folder.rglob("*") if path.is_file())
                shutil.rmtree(claim_folder)
        except (ValueError, OSError) as exc:
            file_errors.append(str(exc))

        # Clean up only paths that are proven to be inside the deleted claim folder.
        for document in documents:
            raw_path = document.get("stored_path")
            if not raw_path:
                continue
            path = Path(str(raw_path)).resolve()
            try:
                path.relative_to(claim_folder)
            except ValueError:
                continue
            if path.exists():
                try:
                    path.unlink()
                    removed_files += 1
                except OSError as exc:
                    file_errors.append(f"{path}: {exc}")

    return {
        "claim_no": claim.get("claim_no"),
        "patient_name": claim.get("patient_name"),
        "record_counts": counts,
        "affected_claim_ids": affected_claim_ids,
        "removed_files": removed_files,
        "file_errors": file_errors,
    }


def update_claim(claim_id: int, **fields: Any) -> None:
    allowed = {
        "application_no", "patient_name", "policy_no", "hospital", "diagnosis",
        "treatment_start", "treatment_end", "claimed_amount", "supported_amount",
        "recommended_amount", "risk_score", "risk_band", "status", "assigned_to",
    }
    clean = {k: v for k, v in fields.items() if k in allowed}
    if not clean:
        return
    clean["updated_at"] = utc_now()
    sets = ", ".join(f"{k} = ?" for k in clean)
    with connect() as conn:
        conn.execute(f"UPDATE claims SET {sets} WHERE id = ?", (*clean.values(), claim_id))
    audit(claim_id, "system", "CLAIM_UPDATED", clean)


def add_document(claim_id: int, data: dict[str, Any]) -> int:
    with connect() as conn:
        cur = conn.execute(
            """
            INSERT INTO documents (
                claim_id, filename, stored_path, sha256, phash, doc_type,
                ocr_text, ocr_confidence, extracted_json, created_at
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                claim_id, data["filename"], data["stored_path"], data.get("sha256"),
                data.get("phash"), data.get("doc_type", "unknown"), data.get("ocr_text", ""),
                float(data.get("ocr_confidence") or 0),
                json.dumps(data.get("extracted", {}), ensure_ascii=False), utc_now(),
            ),
        )
        document_id = int(cur.lastrowid)
    audit(claim_id, "system", "DOCUMENT_ADDED", {"document_id": document_id, "filename": data["filename"]})
    return document_id


def list_documents(claim_id: int) -> list[dict[str, Any]]:
    with connect() as conn:
        rows = conn.execute("SELECT * FROM documents WHERE claim_id = ? ORDER BY id", (claim_id,)).fetchall()
    out = []
    for r in rows:
        item = dict(r)
        item["extracted"] = json.loads(item.pop("extracted_json") or "{}")
        out.append(item)
    return out


def list_all_documents() -> list[dict[str, Any]]:
    with connect() as conn:
        rows = conn.execute("SELECT * FROM documents ORDER BY id").fetchall()
    out = []
    for r in rows:
        item = dict(r)
        item["extracted"] = json.loads(item.pop("extracted_json") or "{}")
        out.append(item)
    return out



def update_document_analysis(document_id: int, *, doc_type: str | None = None, extracted: dict[str, Any] | None = None) -> None:
    fields: dict[str, Any] = {}
    if doc_type:
        fields["doc_type"] = doc_type
    if extracted is not None:
        fields["extracted_json"] = json.dumps(extracted, ensure_ascii=False)
    if not fields:
        return
    sets = ", ".join(f"{key} = ?" for key in fields)
    with connect() as conn:
        row = conn.execute("SELECT claim_id, filename FROM documents WHERE id = ?", (document_id,)).fetchone()
        if not row:
            raise ValueError("Document not found")
        conn.execute(f"UPDATE documents SET {sets} WHERE id = ?", (*fields.values(), document_id))
    audit(int(row["claim_id"]), "local_ai", "DOCUMENT_AI_ANALYSIS_UPDATED", {"document_id": document_id, "filename": row["filename"], "doc_type": doc_type})

def replace_duplicate_matches(claim_id: int, matches: Iterable[dict[str, Any]]) -> None:
    with connect() as conn:
        conn.execute("DELETE FROM duplicate_matches WHERE claim_id = ?", (claim_id,))
        for m in matches:
            conn.execute(
                """INSERT INTO duplicate_matches
                (claim_id, matched_claim_id, matched_document_id, score, status, reasons_json, created_at)
                VALUES (?, ?, ?, ?, ?, ?, ?)""",
                (
                    claim_id, m.get("matched_claim_id"), m.get("matched_document_id"),
                    float(m["score"]), m["status"], json.dumps(m["reasons"], ensure_ascii=False), utc_now(),
                ),
            )


def list_duplicate_matches(claim_id: int) -> list[dict[str, Any]]:
    with connect() as conn:
        rows = conn.execute("SELECT * FROM duplicate_matches WHERE claim_id = ? ORDER BY score DESC", (claim_id,)).fetchall()
    out = []
    for r in rows:
        x = dict(r)
        x["reasons"] = json.loads(x.pop("reasons_json"))
        out.append(x)
    return out


def replace_clinical_checks(claim_id: int, checks: Iterable[dict[str, Any]]) -> None:
    with connect() as conn:
        conn.execute("DELETE FROM clinical_checks WHERE claim_id = ?", (claim_id,))
        for c in checks:
            conn.execute(
                """INSERT INTO clinical_checks
                (claim_id, check_code, severity, result, explanation, evidence_json, created_at)
                VALUES (?, ?, ?, ?, ?, ?, ?)""",
                (
                    claim_id, c["check_code"], c["severity"], c["result"], c["explanation"],
                    json.dumps(c.get("evidence", []), ensure_ascii=False), utc_now(),
                ),
            )


def list_clinical_checks(claim_id: int) -> list[dict[str, Any]]:
    with connect() as conn:
        rows = conn.execute("SELECT * FROM clinical_checks WHERE claim_id = ? ORDER BY id", (claim_id,)).fetchall()
    out = []
    for r in rows:
        x = dict(r)
        x["evidence"] = json.loads(x.pop("evidence_json") or "[]")
        out.append(x)
    return out


def propose_correction(claim_id: int, field_name: str, current_value: Any, proposed_value: Any,
                       reason: str, document_id: int | None = None, requested_by: str = "chat_user") -> int:
    with connect() as conn:
        cur = conn.execute(
            """INSERT INTO correction_proposals
            (claim_id, document_id, field_name, current_value, proposed_value, reason, requested_by, created_at)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?)""",
            (claim_id, document_id, field_name, str(current_value), str(proposed_value), reason, requested_by, utc_now()),
        )
        proposal_id = int(cur.lastrowid)
    audit(claim_id, requested_by, "CORRECTION_PROPOSED", {"proposal_id": proposal_id, "field": field_name, "from": current_value, "to": proposed_value})
    return proposal_id


def list_corrections(claim_id: int) -> list[dict[str, Any]]:
    with connect() as conn:
        rows = conn.execute("SELECT * FROM correction_proposals WHERE claim_id = ? ORDER BY id DESC", (claim_id,)).fetchall()
    return [dict(r) for r in rows]


def review_correction(proposal_id: int, approve: bool, reviewer: str) -> None:
    with connect() as conn:
        row = conn.execute("SELECT * FROM correction_proposals WHERE id = ?", (proposal_id,)).fetchone()
        if not row:
            raise ValueError("Correction proposal not found")
        status = "Approved" if approve else "Rejected"
        conn.execute(
            "UPDATE correction_proposals SET status = ?, reviewed_by = ?, reviewed_at = ? WHERE id = ?",
            (status, reviewer, utc_now(), proposal_id),
        )
    if approve and row["field_name"] in {"patient_name", "policy_no", "hospital", "diagnosis", "claimed_amount", "supported_amount", "recommended_amount", "status"}:
        value: Any = row["proposed_value"]
        if row["field_name"] in {"claimed_amount", "supported_amount", "recommended_amount"}:
            value = float(value)
        update_claim(int(row["claim_id"]), **{row["field_name"]: value})
    audit(int(row["claim_id"]), reviewer, "CORRECTION_REVIEWED", {"proposal_id": proposal_id, "status": status})


def list_audit(claim_id: int) -> list[dict[str, Any]]:
    with connect() as conn:
        rows = conn.execute("SELECT * FROM audit_events WHERE claim_id = ? ORDER BY id DESC", (claim_id,)).fetchall()
    out = []
    for r in rows:
        x = dict(r)
        x["details"] = json.loads(x.pop("details_json") or "{}")
        out.append(x)
    return out


def get_clinical_inputs(claim_id: int) -> dict[str, Any]:
    with connect() as conn:
        row = conn.execute("SELECT * FROM claim_clinical_inputs WHERE claim_id = ?", (claim_id,)).fetchone()
    if not row:
        return {
            "claim_id": claim_id,
            "diagnosis_text": "",
            "prescribed_medicines": [],
            "billed_medicines": [],
            "reviewer_notes": "",
            "updated_by": None,
            "updated_at": None,
        }
    item = dict(row)
    item["prescribed_medicines"] = json.loads(item.pop("prescribed_medicines_json") or "[]")
    item["billed_medicines"] = json.loads(item.pop("billed_medicines_json") or "[]")
    return item


def save_clinical_inputs(
    claim_id: int,
    diagnosis_text: str,
    prescribed_medicines: list[str],
    billed_medicines: list[str],
    reviewer_notes: str = "",
    updated_by: str = "claim_reviewer",
) -> None:
    now = utc_now()
    clean_prescribed = list(dict.fromkeys(x.strip() for x in prescribed_medicines if x and x.strip()))
    clean_billed = list(dict.fromkeys(x.strip() for x in billed_medicines if x and x.strip()))
    with connect() as conn:
        conn.execute(
            """
            INSERT INTO claim_clinical_inputs
                (claim_id, diagnosis_text, prescribed_medicines_json, billed_medicines_json, reviewer_notes, updated_by, updated_at)
            VALUES (?, ?, ?, ?, ?, ?, ?)
            ON CONFLICT(claim_id) DO UPDATE SET
                diagnosis_text = excluded.diagnosis_text,
                prescribed_medicines_json = excluded.prescribed_medicines_json,
                billed_medicines_json = excluded.billed_medicines_json,
                reviewer_notes = excluded.reviewer_notes,
                updated_by = excluded.updated_by,
                updated_at = excluded.updated_at
            """,
            (
                claim_id,
                diagnosis_text.strip(),
                json.dumps(clean_prescribed, ensure_ascii=False),
                json.dumps(clean_billed, ensure_ascii=False),
                reviewer_notes.strip(),
                updated_by,
                now,
            ),
        )
    audit(claim_id, updated_by, "CLINICAL_INPUTS_SAVED", {
        "diagnosis_text": diagnosis_text.strip(),
        "prescribed_medicines": clean_prescribed,
        "billed_medicines": clean_billed,
        "reviewer_notes": reviewer_notes.strip(),
    })
