"""Unit tests for the Gmail sync checkpoint used to avoid duplicate ingestion."""

from __future__ import annotations

from datetime import datetime, timezone

from src.sync_state import GmailSyncState, load_gmail_sync_state, save_gmail_sync_state


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
