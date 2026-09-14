"""Temporal search and retrieval over the email knowledge graph."""

from __future__ import annotations

from datetime import datetime, timezone

from graphiti_core.edges import EntityEdge
from graphiti_core.nodes import EpisodicNode

from src.graphiti_engine import GraphitiEngine

# Large enough to cover a full thread's episode history in practice.
_MAX_THREAD_EPISODES = 1000


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
