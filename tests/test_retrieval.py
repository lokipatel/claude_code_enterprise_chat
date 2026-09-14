"""Unit tests for source-classified (email vs. documentation) fact retrieval."""

from __future__ import annotations

from datetime import datetime, timezone
from unittest.mock import AsyncMock

import pytest
from graphiti_core.edges import EntityEdge
from graphiti_core.nodes import EpisodeType, EpisodicNode

from src.config import get_settings
from src.graphiti_engine import GraphitiEngine
from src.retrieval import search_grouped_facts


@pytest.fixture
def engine(mocker) -> GraphitiEngine:
    mock_graphiti_cls = mocker.patch("src.graphiti_engine.Graphiti")
    mock_instance = mock_graphiti_cls.return_value
    mock_instance.search = AsyncMock()
    mock_instance.close = AsyncMock()
    return GraphitiEngine(settings=get_settings())


def _edge(fact: str, episode_uuid: str) -> EntityEdge:
    return EntityEdge(
        group_id="g",
        source_node_uuid="a",
        target_node_uuid="b",
        created_at=datetime.now(timezone.utc),
        name="RELATES_TO",
        fact=fact,
        episodes=[episode_uuid],
    )


def _episode(uuid: str, name: str, source_description: str) -> EpisodicNode:
    return EpisodicNode(
        uuid=uuid,
        name=name,
        group_id="g",
        source=EpisodeType.text,
        source_description=source_description,
        content="irrelevant",
        valid_at=datetime.now(timezone.utc),
    )


class TestSearchGroupedFacts:
    @pytest.mark.asyncio
    async def test_no_edges_returns_empty_groups(self, engine):
        engine.graphiti.search.return_value = []

        grouped = await search_grouped_facts(engine, "anything")

        assert grouped.email_facts == []
        assert grouped.document_facts == []

    @pytest.mark.asyncio
    async def test_classifies_email_and_document_facts_and_extracts_doc_name(
        self, engine, mocker
    ):
        engine.graphiti.search.return_value = [
            _edge("Meeting is at 4 PM.", "ep-email"),
            _edge("Jane Doe is the Product Owner.", "ep-doc"),
        ]
        mocker.patch(
            "src.retrieval.EpisodicNode.get_by_uuids",
            new=AsyncMock(
                return_value=[
                    _episode("ep-email", "Email: Reschedule (msg-1)", "email"),
                    _episode(
                        "ep-doc", "Document: project_charter.docx (docx)", "document:docx"
                    ),
                ]
            ),
        )

        grouped = await search_grouped_facts(engine, "status update")

        assert grouped.email_facts == ["Meeting is at 4 PM."]
        assert len(grouped.document_facts) == 1
        assert grouped.document_facts[0].fact == "Jane Doe is the Product Owner."
        assert grouped.document_facts[0].document_name == "project_charter.docx"

    @pytest.mark.asyncio
    async def test_only_document_facts_leaves_email_facts_empty(self, engine, mocker):
        engine.graphiti.search.return_value = [_edge("Uses Neo4j.", "ep-doc")]
        mocker.patch(
            "src.retrieval.EpisodicNode.get_by_uuids",
            new=AsyncMock(
                return_value=[
                    _episode("ep-doc", "Document: neo4j_schema.sql (sql)", "document:sql"),
                ]
            ),
        )

        grouped = await search_grouped_facts(engine, "what database")

        assert grouped.email_facts == []
        assert grouped.document_facts[0].document_name == "neo4j_schema.sql"
