"""Local checkpoint tracking which Gmail messages have already been ingested.

Graphiti has no built-in notion of "this email was already added" — calling
add_episode() twice for the same message just creates a second episode. This
checkpoint is what lets `sync-gmail` be safely re-run without duplicating
facts for emails it has already seen.
"""

from __future__ import annotations

from datetime import datetime
from pathlib import Path

from pydantic import BaseModel, Field


class GmailSyncState(BaseModel):
    """Which Gmail message IDs have already been ingested into Graphiti."""

    ingested_message_ids: set[str] = Field(default_factory=set)
    last_synced_at: datetime | None = None


def load_gmail_sync_state(path: Path) -> GmailSyncState:
    """Load the checkpoint from disk, or return an empty one if none exists yet."""
    if not path.exists():
        return GmailSyncState()
    return GmailSyncState.model_validate_json(path.read_text(encoding="utf-8"))


def save_gmail_sync_state(state: GmailSyncState, path: Path) -> None:
    """Persist the checkpoint to disk."""
    path.write_text(state.model_dump_json(indent=2), encoding="utf-8")
