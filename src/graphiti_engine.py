"""Graphiti knowledge-graph engine: indices, and email episode ingestion."""

from __future__ import annotations

import logging

from graphiti_core import Graphiti
from graphiti_core.driver.neo4j_driver import Neo4jDriver
from graphiti_core.embedder.openai import OpenAIEmbedder, OpenAIEmbedderConfig
from graphiti_core.graphiti import AddEpisodeResults
from graphiti_core.llm_client import LLMConfig, OpenAIClient
from graphiti_core.nodes import EpisodeType

from src.config import Settings, get_settings
from src.document_loader import DocumentRecord
from src.gmail_client import EmailMessage

logger = logging.getLogger(__name__)

# Shared group_id for all project documents (as opposed to emails, which are
# grouped per-thread). Keeping documents in one group lets their entities
# resolve against each other; graphiti.search() with no group filter still
# surfaces facts from this group alongside email facts.
DOCUMENT_GROUP_ID = "project_documents"


def _format_episode_body(email_data: EmailMessage) -> str:
    """Render an email into a clean text block for Graphiti extraction."""
    return (
        f"From: {email_data.sender}\n"
        f"To: {email_data.recipient or 'unknown'}\n"
        f"Date: {email_data.date.isoformat()}\n"
        f"Subject: {email_data.subject}\n\n"
        f"{email_data.body_text}"
    )


class GraphitiEngine:
    """Wraps a Graphiti instance configured from application settings."""

    def __init__(self, settings: Settings | None = None) -> None:
        self._settings = settings or get_settings()

        llm_config = LLMConfig(
            api_key=self._settings.openai_api_key,
            model=self._settings.openai_llm_model,
            small_model=self._settings.openai_small_llm_model,
        )
        embedder_config = OpenAIEmbedderConfig(
            api_key=self._settings.openai_api_key,
            embedding_model=self._settings.openai_embedding_model,
        )

        # Graphiti's uri/user/password shortcut always opens the driver's
        # default database ("neo4j"), ignoring NEO4J_DATABASE. Build the
        # driver explicitly so our configured database name is honored.
        graph_driver = Neo4jDriver(
            uri=self._settings.neo4j_uri,
            user=self._settings.neo4j_username,
            password=self._settings.neo4j_password,
            database=self._settings.neo4j_database,
        )

        self.graphiti = Graphiti(
            graph_driver=graph_driver,
            llm_client=OpenAIClient(config=llm_config),
            embedder=OpenAIEmbedder(config=embedder_config),
        )

    async def init_indices(self) -> None:
        """Create Graphiti's required Neo4j indices and constraints."""
        await self.graphiti.build_indices_and_constraints()
        logger.info("Graphiti indices and constraints are ready.")

    async def ingest_email_message(self, email_data: EmailMessage) -> AddEpisodeResults:
        """Add a single email as a Graphiti episode, grouped by its thread."""
        episode_name = f"Email: {email_data.subject} ({email_data.message_id})"
        episode_body = _format_episode_body(email_data)

        result = await self.graphiti.add_episode(
            name=episode_name,
            episode_body=episode_body,
            source_description="email",
            reference_time=email_data.date,
            source=EpisodeType.message,
            group_id=email_data.thread_id,
        )
        logger.info(
            "Ingested episode %s into thread %s (%d edges).",
            episode_name,
            email_data.thread_id,
            len(result.edges),
        )
        return result

    async def ingest_document(self, doc: DocumentRecord) -> AddEpisodeResults:
        """Add a project document (code, docs, image, etc.) as a Graphiti episode."""
        episode_name = f"Document: {doc.filename} ({doc.doc_type})"
        episode_body = f"Document: {doc.filename}\nType: {doc.doc_type}\n\n{doc.content_text}"

        result = await self.graphiti.add_episode(
            name=episode_name,
            episode_body=episode_body,
            source_description=f"document:{doc.doc_type}",
            reference_time=doc.modified_at,
            source=EpisodeType.text,
            group_id=DOCUMENT_GROUP_ID,
        )
        logger.info(
            "Ingested document %s (%s, %d edges).",
            doc.filename,
            doc.doc_type,
            len(result.edges),
        )
        return result

    async def close(self) -> None:
        await self.graphiti.close()
