"""Local checkpoints tracking what's already been ingested into Graphiti.

Graphiti has no built-in notion of "this was already added" — calling
add_episode() twice for the same content just creates a second episode.
These checkpoints are what let `sync-gmail` and `sync-docs` be safely
re-run without duplicating facts for things they've already seen.
"""

from __future__ import annotations

from datetime import datetime
from pathlib import Path
from typing import TypeVar

from pydantic import BaseModel, Field

_StateT = TypeVar("_StateT", bound=BaseModel)


def _load_state(model: type[_StateT], path: Path) -> _StateT:
    if not path.exists():
        return model()
    return model.model_validate_json(path.read_text(encoding="utf-8"))


def _save_state(state: BaseModel, path: Path) -> None:
    path.write_text(state.model_dump_json(indent=2), encoding="utf-8")


class GmailSyncState(BaseModel):
    """Which Gmail message IDs have already been ingested into Graphiti."""

    ingested_message_ids: set[str] = Field(default_factory=set)
    last_synced_at: datetime | None = None


def load_gmail_sync_state(path: Path) -> GmailSyncState:
    """Load the checkpoint from disk, or return an empty one if none exists yet."""
    return _load_state(GmailSyncState, path)


def save_gmail_sync_state(state: GmailSyncState, path: Path) -> None:
    """Persist the checkpoint to disk."""
    _save_state(state, path)


class DocSyncState(BaseModel):
    """Maps each document's doc_id to the content hash last ingested for it.

    Unlike emails, documents are mutable files: hashing content_text (rather
    than just tracking doc_id) means an edited document gets re-ingested on
    the next sync-docs run, while an unchanged one is skipped.
    """

    ingested_documents: dict[str, str] = Field(default_factory=dict)
    last_synced_at: datetime | None = None


def load_doc_sync_state(path: Path) -> DocSyncState:
    """Load the checkpoint from disk, or return an empty one if none exists yet."""
    return _load_state(DocSyncState, path)


def save_doc_sync_state(state: DocSyncState, path: Path) -> None:
    """Persist the checkpoint to disk."""
    _save_state(state, path)
