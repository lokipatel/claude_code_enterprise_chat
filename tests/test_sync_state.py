"""Unit tests for the sync checkpoints used to avoid duplicate ingestion."""

from __future__ import annotations

import hashlib
from datetime import datetime, timezone

from src.sync_state import (
    DocSyncState,
    GmailSyncState,
    load_doc_sync_state,
    load_gmail_sync_state,
    save_doc_sync_state,
    save_gmail_sync_state,
)


def _hash(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


class TestGmailSyncState:
    def test_missing_checkpoint_file_yields_empty_state(self, tmp_path):
        state = load_gmail_sync_state(tmp_path / "does_not_exist.json")

        assert state.ingested_message_ids == set()
        assert state.last_synced_at is None

    def test_save_then_load_round_trips(self, tmp_path):
        path = tmp_path / "sync_state.json"
        state = GmailSyncState(
            ingested_message_ids={"msg-1", "msg-2"},
            last_synced_at=datetime(2026, 1, 1, tzinfo=timezone.utc),
        )

        save_gmail_sync_state(state, path)
        reloaded = load_gmail_sync_state(path)

        assert reloaded.ingested_message_ids == {"msg-1", "msg-2"}
        assert reloaded.last_synced_at == datetime(2026, 1, 1, tzinfo=timezone.utc)


class TestDedupLogic:
    def test_already_ingested_ids_are_excluded_from_new_messages(self):
        """Mirrors the filtering src/cli.py's sync-gmail command applies."""
        state = GmailSyncState(ingested_message_ids={"msg-1", "msg-2"})
        raw_messages = [{"id": "msg-1"}, {"id": "msg-2"}, {"id": "msg-3"}]

        new_messages = [m for m in raw_messages if m["id"] not in state.ingested_message_ids]

        assert [m["id"] for m in new_messages] == ["msg-3"]


class TestDocSyncState:
    def test_missing_checkpoint_file_yields_empty_state(self, tmp_path):
        state = load_doc_sync_state(tmp_path / "does_not_exist.json")

        assert state.ingested_documents == {}
        assert state.last_synced_at is None

    def test_save_then_load_round_trips(self, tmp_path):
        path = tmp_path / "doc_sync_state.json"
        state = DocSyncState(
            ingested_documents={"schema/neo4j_schema.sql": _hash("CREATE TABLE ...")},
            last_synced_at=datetime(2026, 1, 1, tzinfo=timezone.utc),
        )

        save_doc_sync_state(state, path)
        reloaded = load_doc_sync_state(path)

        assert reloaded.ingested_documents == {
            "schema/neo4j_schema.sql": _hash("CREATE TABLE ...")
        }
        assert reloaded.last_synced_at == datetime(2026, 1, 1, tzinfo=timezone.utc)


class TestDocDedupLogic:
    """Mirrors the new-or-changed filtering src/cli.py's sync-docs command applies."""

    def test_unchanged_document_is_skipped(self):
        state = DocSyncState(ingested_documents={"notes.txt": _hash("hello")})

        content_hash = _hash("hello")
        is_new_or_changed = state.ingested_documents.get("notes.txt") != content_hash

        assert is_new_or_changed is False

    def test_edited_document_is_treated_as_changed(self):
        state = DocSyncState(ingested_documents={"notes.txt": _hash("hello")})

        content_hash = _hash("hello, but edited")
        is_new_or_changed = state.ingested_documents.get("notes.txt") != content_hash

        assert is_new_or_changed is True

    def test_unseen_document_is_new(self):
        state = DocSyncState(ingested_documents={})

        content_hash = _hash("brand new content")
        is_new_or_changed = state.ingested_documents.get("new_doc.txt") != content_hash

        assert is_new_or_changed is True
