"""Temporal search and retrieval over the email knowledge graph."""

from __future__ import annotations

import re
from datetime import datetime, timezone

from graphiti_core.edges import EntityEdge
from graphiti_core.nodes import EpisodicNode
from pydantic import BaseModel, Field

from src.graphiti_engine import GraphitiEngine

# Large enough to cover a full thread's episode history in practice.
_MAX_THREAD_EPISODES = 1000

# Matches episode names of the form "Document: <filename> (<doc_type>)",
# set by GraphitiEngine.ingest_document.
_DOCUMENT_EPISODE_NAME_RE = re.compile(r"^Document: (.+) \([^()]+\)$")


async def search_current_facts(
    engine: GraphitiEngine, query: str, num_results: int = 10
) -> list[EntityEdge]:
    """Hybrid search for currently-valid facts relevant to `query`.

    Graphiti's `search` already restricts results to edges that have not
    been invalidated, so superseded facts (e.g. an old meeting time) are
    excluded automatically.
    """
    return await engine.graphiti.search(query=query, num_results=num_results)


async def get_thread_history(
    engine: GraphitiEngine, thread_id: str
) -> list[EpisodicNode]:
    """Return all episodes ingested for a given email thread, oldest first.

    Each episode's `valid_at`/edges reflect Graphiti's temporal state
    transitions, so this also exposes when facts were superseded.
    """
    episodes = await engine.graphiti.retrieve_episodes(
        reference_time=datetime.now(timezone.utc),
        last_n=_MAX_THREAD_EPISODES,
        group_ids=[thread_id],
    )
    return sorted(episodes, key=lambda episode: episode.valid_at)


def format_facts(edges: list[EntityEdge]) -> list[str]:
    """Render entity-edge facts as plain strings for CLI display."""
    return [edge.fact for edge in edges]


class DocumentFact(BaseModel):
    """A fact traced back to the specific project document it came from."""

    fact: str
    document_name: str


class GroupedFacts(BaseModel):
    """Search results split by source, so callers can tell email vs. documentation apart."""

    email_facts: list[str] = Field(default_factory=list)
    document_facts: list[DocumentFact] = Field(default_factory=list)


def _extract_document_name(episode_name: str) -> str:
    match = _DOCUMENT_EPISODE_NAME_RE.match(episode_name)
    return match.group(1) if match else episode_name


async def search_grouped_facts(
    engine: GraphitiEngine, query: str, num_results: int = 10
) -> GroupedFacts:
    """Hybrid search that also classifies each fact as coming from an email
    thread or a project document, tracing document facts back to the
    specific filename they were extracted from.
    """
    edges = await engine.graphiti.search(query=query, num_results=num_results)
    if not edges:
        return GroupedFacts()

    episode_uuids = list({uuid for edge in edges for uuid in edge.episodes})
    episodes = await EpisodicNode.get_by_uuids(engine.graphiti.driver, episode_uuids)
    episodes_by_uuid = {episode.uuid: episode for episode in episodes}

    grouped = GroupedFacts()
    for edge in edges:
        episode = next(
            (episodes_by_uuid[uuid] for uuid in edge.episodes if uuid in episodes_by_uuid),
            None,
        )
        if episode is None:
            continue

        if episode.source_description == "email":
            grouped.email_facts.append(edge.fact)
        elif episode.source_description.startswith("document:"):
            grouped.document_facts.append(
                DocumentFact(
                    fact=edge.fact,
                    document_name=_extract_document_name(episode.name),
                )
            )

    return grouped
