"""Unit tests for turning raw graph facts into a short plain-English answer."""

from __future__ import annotations

from datetime import datetime, timezone
from unittest.mock import AsyncMock, MagicMock

import pytest
from graphiti_core.edges import EntityEdge

from src.config import get_settings
from src.graphiti_engine import GraphitiEngine
from src.qa import NO_FACTS_ANSWER, answer_question


@pytest.fixture
def engine(mocker) -> GraphitiEngine:
    mock_graphiti_cls = mocker.patch("src.graphiti_engine.Graphiti")
    mock_instance = mock_graphiti_cls.return_value
    mock_instance.search = AsyncMock()
    mock_instance.close = AsyncMock()
    return GraphitiEngine(settings=get_settings())


def _make_edge(fact: str) -> EntityEdge:
    return EntityEdge(
        group_id="test",
        source_node_uuid="a",
        target_node_uuid="b",
        created_at=datetime.now(timezone.utc),
        name="RELATES_TO",
        fact=fact,
    )


class TestAnswerQuestion:
    @pytest.mark.asyncio
    async def test_returns_canned_message_when_no_facts_found(self, engine, mocker):
        engine.graphiti.search.return_value = []
        openai_cls = mocker.patch("src.qa.AsyncOpenAI")

        answer = await answer_question(engine, "What time is the meeting?")

        assert answer == NO_FACTS_ANSWER
        openai_cls.assert_not_called()  # no facts -> skip the LLM call entirely

    @pytest.mark.asyncio
    async def test_synthesizes_answer_from_facts_via_openai(self, engine, mocker):
        engine.graphiti.search.return_value = [_make_edge("The meeting is at 4 PM.")]

        mock_response = MagicMock()
        mock_response.choices = [MagicMock(message=MagicMock(content="  It's at 4 PM.  "))]
        mock_client = MagicMock()
        mock_client.chat.completions.create = AsyncMock(return_value=mock_response)
        mocker.patch("src.qa.AsyncOpenAI", return_value=mock_client)

        answer = await answer_question(engine, "What time is the meeting?")

        assert answer == "It's at 4 PM."
        _, kwargs = mock_client.chat.completions.create.await_args
        assert "The meeting is at 4 PM." in kwargs["messages"][1]["content"]
        assert "What time is the meeting?" in kwargs["messages"][1]["content"]
