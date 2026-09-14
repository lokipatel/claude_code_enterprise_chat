"""Typer CLI for the Email Knowledge Graph system.

Usage:
    python -m src.cli init
    python -m src.cli sync-gmail --max 10
    python -m src.cli query "What is the updated meeting time with Jane?"
"""

from __future__ import annotations

import asyncio
import hashlib
import logging
from datetime import datetime, timezone
from pathlib import Path

import typer

from src.config import get_settings
from src.document_loader import load_documents_dir
from src.gmail_client import GmailAuthError, GmailClient
from src.graphiti_engine import GraphitiEngine
from src.qa import answer_question
from src.sync_state import (
    DocSyncState,
    GmailSyncState,
    load_doc_sync_state,
    load_gmail_sync_state,
    save_doc_sync_state,
    save_gmail_sync_state,
)

# WARNING by default so third-party libraries (httpx, neo4j, graphiti_core,
# googleapiclient) stay quiet; the CLI's own typer.echo() progress lines and
# final answers are what the user should actually see.
logging.basicConfig(level=logging.WARNING, format="%(levelname)s %(name)s: %(message)s")
logger = logging.getLogger(__name__)

app = typer.Typer(help="Email Knowledge Graph CLI (Gmail + Graphiti + Neo4j).")


@app.command()
def init() -> None:
    """Create Graphiti's required Neo4j indices and constraints."""

    async def _run() -> None:
        engine = GraphitiEngine()
        try:
            await engine.init_indices()
            typer.echo("Neo4j indices and constraints are ready.")
        finally:
            await engine.close()

    asyncio.run(_run())


@app.command("sync-gmail")
def sync_gmail(
    max: int = typer.Option(10, "--max", help="Max number of emails to fetch per run."),
    query: str = typer.Option("label:INBOX", "--query", help="Gmail search query."),
    reset: bool = typer.Option(
        False, "--reset", help="Ignore the checkpoint and re-ingest every fetched email."
    ),
) -> None:
    """Authenticate with Gmail, pull recent emails, and ingest only the ones not yet synced.

    Ingested message IDs are checkpointed to sync_state.json, so re-running this
    command won't create duplicate episodes for emails already in the graph.
    Note --max bounds each run's fetch window: if more than `max` new emails
    arrived since the last sync, only the most recent `max` are considered.
    """

    async def _run() -> None:
        settings = get_settings()
        state = GmailSyncState() if reset else load_gmail_sync_state(settings.sync_state_path)

        gmail = GmailClient(settings)
        try:
            raw_messages = gmail.fetch_latest_emails(max_results=max, query=query)
        except GmailAuthError as exc:
            typer.echo(f"Gmail authentication failed: {exc}", err=True)
            raise typer.Exit(code=1) from exc

        new_messages = [m for m in raw_messages if m["id"] not in state.ingested_message_ids]
        already_synced = len(raw_messages) - len(new_messages)

        if not new_messages:
            typer.echo(f"No new emails to ingest ({already_synced} already synced).")
            return

        engine = GraphitiEngine(settings)
        try:
            for raw in new_messages:
                email_data = GmailClient.parse_email_payload(raw)
                await engine.ingest_email_message(email_data)
                state.ingested_message_ids.add(email_data.message_id)
                typer.echo(f"Ingested: {email_data.subject!r} ({email_data.message_id})")

            state.last_synced_at = datetime.now(timezone.utc)
            save_gmail_sync_state(state, settings.sync_state_path)
            typer.echo(
                f"Synced {len(new_messages)} new email(s) into Graphiti "
                f"({already_synced} already up to date)."
            )
        finally:
            await engine.close()

    asyncio.run(_run())


def _content_hash(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


@app.command("sync-docs")
def sync_docs(
    path: str = typer.Option(
        "documents", "--path", help="Directory of project documents to ingest."
    ),
    reset: bool = typer.Option(
        False, "--reset", help="Ignore the checkpoint and re-ingest every document."
    ),
) -> None:
    """Load project documents and ingest only the ones that are new or changed.

    Ingested content hashes are checkpointed to doc_sync_state.json (keyed by
    doc_id, not just filename), so re-running this command won't create
    duplicate episodes for unchanged documents -- but editing a document's
    content DOES cause it to be re-ingested on the next run.
    """

    async def _run() -> None:
        settings = get_settings()
        docs_dir = Path(path)
        if not docs_dir.is_dir():
            typer.echo(f"Documents directory not found: {docs_dir}", err=True)
            raise typer.Exit(code=1)

        docs = load_documents_dir(docs_dir)
        if not docs:
            typer.echo(f"No documents found under {docs_dir}.")
            return

        state = DocSyncState() if reset else load_doc_sync_state(settings.doc_sync_state_path)

        new_or_changed = []
        for doc in docs:
            content_hash = _content_hash(doc.content_text)
            if state.ingested_documents.get(doc.doc_id) != content_hash:
                new_or_changed.append((doc, content_hash))
        already_synced = len(docs) - len(new_or_changed)

        if not new_or_changed:
            typer.echo(f"No new or changed documents to ingest ({already_synced} already up to date).")
            return

        engine = GraphitiEngine(settings)
        try:
            for doc, content_hash in new_or_changed:
                await engine.ingest_document(doc)
                state.ingested_documents[doc.doc_id] = content_hash
                typer.echo(f"Ingested: {doc.doc_id} ({doc.doc_type}, {len(doc.chunks)} chunk(s))")

            state.last_synced_at = datetime.now(timezone.utc)
            save_doc_sync_state(state, settings.doc_sync_state_path)
            typer.echo(
                f"Synced {len(new_or_changed)} new/changed document(s) into Graphiti "
                f"({already_synced} already up to date)."
            )
        finally:
            await engine.close()

    asyncio.run(_run())


@app.command()
def query(question: str = typer.Argument(..., help="Natural-language question.")) -> None:
    """Ask a question and get a short, plain-English answer from the knowledge graph."""

    async def _run() -> None:
        engine = GraphitiEngine()
        try:
            answer = await answer_question(engine, question)
            typer.echo(answer)
        finally:
            await engine.close()

    asyncio.run(_run())


if __name__ == "__main__":
    app()
