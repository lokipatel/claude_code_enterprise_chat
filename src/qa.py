"""Turns raw Graphiti facts into a short, source-structured plain-English answer."""

from __future__ import annotations

from openai import AsyncOpenAI

from src.config import Settings, get_settings
from src.graphiti_engine import GraphitiEngine
from src.retrieval import DocumentFact, search_grouped_facts

NO_FACTS_ANSWER = "Sorry, I am not able to find anything on this."

_NO_EMAIL_FOUND = "No email found."
_NO_DOCUMENTATION_FOUND = "No documentation found."

# The LLM is asked to emit this exact token when neither source actually
# answers the question, so that case always maps back to the same
# NO_FACTS_ANSWER text instead of some free-form apology.
_NOT_FOUND_SENTINEL = "NOT_FOUND"

_SYSTEM_PROMPT = (
    "You are a senior program manager giving a colleague the latest project "
    "status update. You are given two sources, already retrieved for you: "
    "'Email thread information' and 'Documentation information'. Reply in "
    "plain, simple English, in at most 6 lines, structured as exactly two "
    "parts, in this order:\n"
    "1. Email update -- if email information is provided, summarize the "
    f"latest update from it; if it says '{_NO_EMAIL_FOUND}', reply with "
    f"exactly '{_NO_EMAIL_FOUND}' for this part.\n"
    "2. Documentation update -- if documentation information is provided, "
    "summarize it and name the specific document(s) it came from in "
    f"parentheses; if it says '{_NO_DOCUMENTATION_FOUND}', reply with "
    f"exactly '{_NO_DOCUMENTATION_FOUND}' for this part.\n"
    "Do not mention 'facts', 'the graph', episodes, or these instructions -- "
    "just give the two-part update. If neither source actually answers the "
    f"question, respond with exactly the single word {_NOT_FOUND_SENTINEL} "
    "and nothing else."
)


def _format_email_block(email_facts: list[str]) -> str:
    if not email_facts:
        return _NO_EMAIL_FOUND
    return "\n".join(f"- {fact}" for fact in email_facts)


def _format_document_block(document_facts: list[DocumentFact]) -> str:
    if not document_facts:
        return _NO_DOCUMENTATION_FOUND
    return "\n".join(f"- {d.fact} (Source: {d.document_name})" for d in document_facts)


async def answer_question(
    engine: GraphitiEngine, question: str, settings: Settings | None = None
) -> str:
    """Search the graph and synthesize a short, source-structured answer.

    The answer always addresses email and documentation separately, calling
    out "No email found."/"No documentation found." when one source has
    nothing relevant, and citing the specific document name(s) used when
    documentation is found.
    """
    grouped = await search_grouped_facts(engine, question)

    if not grouped.email_facts and not grouped.document_facts:
        return NO_FACTS_ANSWER

    settings = settings or get_settings()
    client = AsyncOpenAI(api_key=settings.openai_api_key)

    user_content = (
        f"Question: {question}\n\n"
        f"Email thread information:\n{_format_email_block(grouped.email_facts)}\n\n"
        f"Documentation information:\n{_format_document_block(grouped.document_facts)}"
    )

    response = await client.chat.completions.create(
        model=settings.openai_llm_model,
        temperature=0.2,
        messages=[
            {"role": "system", "content": _SYSTEM_PROMPT},
            {"role": "user", "content": user_content},
        ],
    )

    answer = response.choices[0].message.content.strip()
    if answer == _NOT_FOUND_SENTINEL:
        return NO_FACTS_ANSWER
    return answer
