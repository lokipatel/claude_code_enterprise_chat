"""Turns raw Graphiti facts into a short, plain-English answer for the CLI."""

from __future__ import annotations

from openai import AsyncOpenAI

from src.config import Settings, get_settings
from src.graphiti_engine import GraphitiEngine
from src.retrieval import format_facts, search_current_facts

NO_FACTS_ANSWER = "I couldn't find anything relevant to that in the knowledge graph."

_SYSTEM_PROMPT = (
    "Answer the user's question using only the facts given below. Reply in "
    "plain, simple English in at most 4 short lines. Do not mention 'facts', "
    "'the graph', or your sources -- just state the answer directly. If the "
    "facts don't actually answer the question, say so in one short line."
)


async def answer_question(
    engine: GraphitiEngine, question: str, settings: Settings | None = None
) -> str:
    """Search the graph and synthesize a short, direct answer to `question`."""
    edges = await search_current_facts(engine, question)
    facts = format_facts(edges)

    if not facts:
        return NO_FACTS_ANSWER

    settings = settings or get_settings()
    client = AsyncOpenAI(api_key=settings.openai_api_key)
    facts_block = "\n".join(f"- {fact}" for fact in facts)

    response = await client.chat.completions.create(
        model=settings.openai_llm_model,
        temperature=0.2,
        messages=[
            {"role": "system", "content": _SYSTEM_PROMPT},
            {"role": "user", "content": f"Question: {question}\n\nFacts:\n{facts_block}"},
        ],
    )
    return response.choices[0].message.content.strip()
