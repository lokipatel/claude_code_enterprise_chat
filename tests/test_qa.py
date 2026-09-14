"""Unit tests for turning grouped (email/documentation) facts into a short,
source-structured plain-English answer.
"""

from __future__ import annotations

from unittest.mock import AsyncMock, MagicMock

import pytest

from src.config import get_settings
from src.graphiti_engine import GraphitiEngine
from src.qa import NO_FACTS_ANSWER, answer_question
from src.retrieval import DocumentFact, GroupedFacts


@pytest.fixture
def engine(mocker) -> GraphitiEngine:
    mock_graphiti_cls = mocker.patch("src.graphiti_engine.Graphiti")
    mock_instance = mock_graphiti_cls.return_value
    mock_instance.close = AsyncMock()
    return GraphitiEngine(settings=get_settings())


def _mock_openai_returning(mocker, content: str) -> MagicMock:
    mock_response = MagicMock()
    mock_response.choices = [MagicMock(message=MagicMock(content=content))]
    mock_client = MagicMock()
    mock_client.chat.completions.create = AsyncMock(return_value=mock_response)
    mocker.patch("src.qa.AsyncOpenAI", return_value=mock_client)
    return mock_client


class TestAnswerQuestion:
    @pytest.mark.asyncio
    async def test_returns_canned_message_when_nothing_found_in_either_source(
        self, engine, mocker
    ):
        mocker.patch("src.qa.search_grouped_facts", new=AsyncMock(return_value=GroupedFacts()))
        openai_cls = mocker.patch("src.qa.AsyncOpenAI")

        answer = await answer_question(engine, "What time is the meeting?")

        assert answer == NO_FACTS_ANSWER
        openai_cls.assert_not_called()  # nothing found -> skip the LLM call entirely

    @pytest.mark.asyncio
    async def test_passes_structured_email_and_document_blocks_to_the_prompt(
        self, engine, mocker
    ):
        grouped = GroupedFacts(
            email_facts=["The meeting is at 4 PM."],
            document_facts=[
                DocumentFact(fact="Jane Doe is the Product Owner.", document_name="charter.docx")
            ],
        )
        mocker.patch("src.qa.search_grouped_facts", new=AsyncMock(return_value=grouped))
        mock_client = _mock_openai_returning(mocker, "  Summary text.  ")

        answer = await answer_question(engine, "What's the latest update?")

        assert answer == "Summary text."
        _, kwargs = mock_client.chat.completions.create.await_args
        user_content = kwargs["messages"][1]["content"]
        assert "The meeting is at 4 PM." in user_content
        assert "Jane Doe is the Product Owner." in user_content
        assert "charter.docx" in user_content
        assert "No email found." not in user_content
        assert "No documentation found." not in user_content

    @pytest.mark.asyncio
    async def test_marks_missing_email_source_explicitly(self, engine, mocker):
        grouped = GroupedFacts(
            document_facts=[DocumentFact(fact="Uses Neo4j.", document_name="schema.sql")]
        )
        mocker.patch("src.qa.search_grouped_facts", new=AsyncMock(return_value=grouped))
        mock_client = _mock_openai_returning(mocker, "No email found. Uses Neo4j (schema.sql).")

        await answer_question(engine, "what database and any related emails?")

        _, kwargs = mock_client.chat.completions.create.await_args
        user_content = kwargs["messages"][1]["content"]
        assert "Email thread information:\nNo email found." in user_content
        assert "Documentation information:\n- Uses Neo4j. (Source: schema.sql)" in user_content

    @pytest.mark.asyncio
    async def test_marks_missing_documentation_source_explicitly(self, engine, mocker):
        grouped = GroupedFacts(email_facts=["Payment of Rs.505 was made."])
        mocker.patch("src.qa.search_grouped_facts", new=AsyncMock(return_value=grouped))
        mock_client = _mock_openai_returning(mocker, "Payment update. No documentation found.")

        await answer_question(engine, "any payment update?")

        _, kwargs = mock_client.chat.completions.create.await_args
        user_content = kwargs["messages"][1]["content"]
        assert "Documentation information:\nNo documentation found." in user_content

    @pytest.mark.asyncio
    async def test_irrelevant_sources_map_to_the_same_apology(self, engine, mocker):
        """Both sources had content, but neither answers the question -> the
        exact same apology text as the nothing-found case, not a free-form one."""
        grouped = GroupedFacts(
            email_facts=["Unrelated fact."],
            document_facts=[DocumentFact(fact="Also unrelated.", document_name="doc.txt")],
        )
        mocker.patch("src.qa.search_grouped_facts", new=AsyncMock(return_value=grouped))
        _mock_openai_returning(mocker, "NOT_FOUND")

        answer = await answer_question(engine, "What is the capital of France?")

        assert answer == NO_FACTS_ANSWER
