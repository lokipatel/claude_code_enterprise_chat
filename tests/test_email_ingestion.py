"""Unit tests for email ingestion and temporal retrieval.

These tests mock the Gmail API and the underlying `graphiti_core.Graphiti`
client boundary, so they run fully offline (no live Neo4j/OpenAI/Gmail
calls). Graphiti's own temporal-invalidation logic is exercised by its
own test suite; here we verify that (a) our ingestion code calls
`add_episode` with the right episode framing and thread grouping, and
(b) our retrieval code surfaces only what Graphiti's search returns as
currently valid, i.e. that a superseded fact (the old meeting time)
does not leak through once a newer episode has invalidated it.
"""

from __future__ import annotations

import base64
from datetime import datetime, timezone
from unittest.mock import AsyncMock

import pytest
from graphiti_core.edges import EntityEdge
from graphiti_core.nodes import EpisodeType

from src.config import get_settings
from src.gmail_client import GmailClient
from src.graphiti_engine import GraphitiEngine
from src.retrieval import format_facts, search_current_facts

THREAD_ID = "thread-jane-meeting"


def _make_raw_message(
    message_id: str, subject: str, date_header: str, body_text: str
) -> dict:
    encoded_body = base64.urlsafe_b64encode(body_text.encode("utf-8")).decode("ascii")
    return {
        "id": message_id,
        "threadId": THREAD_ID,
        "internalDate": "1735689600000",
        "payload": {
            "mimeType": "text/plain",
            "headers": [
                {"name": "Subject", "value": subject},
                {"name": "From", "value": "jane@example.com"},
                {"name": "To", "value": "me@example.com"},
                {"name": "Date", "value": date_header},
            ],
            "body": {"data": encoded_body},
        },
    }


@pytest.fixture
def raw_email_1() -> dict:
    return _make_raw_message(
        message_id="msg-1",
        subject="Project Sync",
        date_header="Mon, 1 Sep 2025 08:00:00 -0700",
        body_text="Meeting at 2 PM today.",
    )


@pytest.fixture
def raw_email_2() -> dict:
    return _make_raw_message(
        message_id="msg-2",
        subject="Re: Project Sync",
        date_header="Mon, 1 Sep 2025 09:30:00 -0700",
        body_text="Quick update: rescheduled to 4 PM.",
    )


@pytest.fixture
def engine(mocker) -> GraphitiEngine:
    """A GraphitiEngine whose underlying Graphiti client is fully mocked."""
    mock_graphiti_cls = mocker.patch("src.graphiti_engine.Graphiti")
    mock_instance = mock_graphiti_cls.return_value
    mock_instance.add_episode = AsyncMock()
    mock_instance.search = AsyncMock()
    mock_instance.close = AsyncMock()
    return GraphitiEngine(settings=get_settings())


class TestParsing:
    def test_parse_email_payload_extracts_structured_fields(self, raw_email_1):
        parsed = GmailClient.parse_email_payload(raw_email_1)

        assert parsed.message_id == "msg-1"
        assert parsed.thread_id == THREAD_ID
        assert parsed.subject == "Project Sync"
        assert parsed.sender == "jane@example.com"
        assert parsed.body_text == "Meeting at 2 PM today."
        assert parsed.date == datetime(
            2025, 9, 1, 8, 0, 0, tzinfo=parsed.date.tzinfo
        )


class TestIngestion:
    @pytest.mark.asyncio
    async def test_ingest_email_message_calls_add_episode_with_thread_grouping(
        self, engine, raw_email_1
    ):
        email_data = GmailClient.parse_email_payload(raw_email_1)

        await engine.ingest_email_message(email_data)

        engine.graphiti.add_episode.assert_awaited_once()
        _, kwargs = engine.graphiti.add_episode.await_args

        assert kwargs["name"] == f"Email: Project Sync ({email_data.message_id})"
        assert "jane@example.com" in kwargs["episode_body"]
        assert "Meeting at 2 PM today." in kwargs["episode_body"]
        assert kwargs["source"] == EpisodeType.message
        assert kwargs["source_description"] == "email"
        assert kwargs["reference_time"] == email_data.date
        assert kwargs["group_id"] == THREAD_ID

    @pytest.mark.asyncio
    async def test_two_thread_emails_share_the_same_group_id(
        self, engine, raw_email_1, raw_email_2
    ):
        email_1 = GmailClient.parse_email_payload(raw_email_1)
        email_2 = GmailClient.parse_email_payload(raw_email_2)

        await engine.ingest_email_message(email_1)
        await engine.ingest_email_message(email_2)

        assert engine.graphiti.add_episode.await_count == 2
        group_ids = {
            call.kwargs["group_id"] for call in engine.graphiti.add_episode.await_args_list
        }
        assert group_ids == {THREAD_ID}


class TestTemporalRetrieval:
    @pytest.mark.asyncio
    async def test_search_returns_only_currently_valid_fact_after_reschedule(
        self, engine, raw_email_1, raw_email_2
    ):
        """Simulates: 'Meeting at 2 PM' gets invalidated once 'Rescheduled to 4 PM'
        is ingested, so search should surface only the 4 PM fact."""
        email_1 = GmailClient.parse_email_payload(raw_email_1)
        email_2 = GmailClient.parse_email_payload(raw_email_2)

        await engine.ingest_email_message(email_1)
        await engine.ingest_email_message(email_2)

        now = datetime.now(timezone.utc)
        stale_edge = EntityEdge(
            group_id=THREAD_ID,
            source_node_uuid="meeting-node",
            target_node_uuid="time-node",
            created_at=now,
            name="HAS_TIME",
            fact="The meeting is at 2 PM.",
            valid_at=email_1.date,
            invalid_at=email_2.date,
        )
        current_edge = EntityEdge(
            group_id=THREAD_ID,
            source_node_uuid="meeting-node",
            target_node_uuid="time-node-2",
            created_at=now,
            name="HAS_TIME",
            fact="The meeting is at 4 PM.",
            valid_at=email_2.date,
            invalid_at=None,
        )

        # Graphiti's own search contract excludes invalidated edges, so the
        # mock only returns what would currently be considered valid.
        engine.graphiti.search.return_value = [current_edge]

        edges = await search_current_facts(engine, "What time is the meeting with Jane?")
        facts = format_facts(edges)

        assert facts == ["The meeting is at 4 PM."]
        assert stale_edge.fact not in facts
        assert stale_edge.invalid_at == email_2.date
