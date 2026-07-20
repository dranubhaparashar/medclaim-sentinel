from pathlib import Path

from medclaim import db


def test_delete_claim_removes_records_files_and_duplicate_links(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    monkeypatch.setattr(db, "DB_PATH", tmp_path / "storage" / "medclaim.db")
    db.init_db()

    deleted_id = db.create_claim({
        "claim_no": "CLM-DELETE-001",
        "patient_name": "Pending OCR Review",
        "status": "Documents Processing",
    })
    linked_id = db.create_claim({
        "claim_no": "CLM-LINKED-001",
        "patient_name": "Linked Patient",
        "status": "Possible Duplicate",
    })

    claim_folder = Path("storage/claims") / str(deleted_id)
    claim_folder.mkdir(parents=True, exist_ok=True)
    document_path = claim_folder / "partial.jpeg"
    document_path.write_bytes(b"partial upload")
    document_id = db.add_document(deleted_id, {
        "filename": "partial.jpeg",
        "stored_path": str(document_path),
        "doc_type": "unknown",
        "extracted": {},
    })
    db.replace_duplicate_matches(linked_id, [{
        "matched_claim_id": deleted_id,
        "matched_document_id": document_id,
        "score": 0.95,
        "status": "probable_duplicate",
        "reasons": ["same partial file"],
    }])
    db.replace_clinical_checks(deleted_id, [{
        "check_code": "TEST",
        "severity": "low",
        "result": "information",
        "explanation": "test",
        "evidence": {},
    }])
    db.save_clinical_inputs(deleted_id, "test", ["A"], ["A"])
    db.propose_correction(deleted_id, "patient_name", "Pending OCR Review", "Test Patient", "test")

    result = db.delete_claim(
        deleted_id,
        deleted_by="tester",
        reason="Incomplete upload",
        delete_files=True,
    )

    assert result["claim_no"] == "CLM-DELETE-001"
    assert linked_id in result["affected_claim_ids"]
    assert db.get_claim(deleted_id) is None
    assert db.list_documents(deleted_id) == []
    assert db.list_clinical_checks(deleted_id) == []
    assert db.list_duplicate_matches(linked_id) == []
    assert not claim_folder.exists()

    with db.connect() as conn:
        deletion = conn.execute(
            "SELECT claim_no, reason, deleted_by FROM claim_deletion_log WHERE claim_no = ?",
            ("CLM-DELETE-001",),
        ).fetchone()
    assert deletion is not None
    assert deletion["reason"] == "Incomplete upload"
    assert deletion["deleted_by"] == "tester"
