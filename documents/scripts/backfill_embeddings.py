"""Mock document: maintenance script for re-embedding stale episodes.

Not wired into the CLI — this simulates an ops runbook script an engineer
would reach for after rotating the OpenAI embedding model, so that older
Graphiti episodes get their fact embeddings refreshed in place.
"""

from __future__ import annotations

import asyncio
import logging

from neo4j import AsyncGraphDatabase

from src.config import get_settings
from src.graphiti_engine import GraphitiEngine

logger = logging.getLogger(__name__)

BATCH_SIZE = 200


async def backfill_stale_embeddings(new_embedding_model: str) -> int:
    """Re-embed every EntityEdge fact using the current embedder.

    Returns the number of edges updated.
    """
    settings = get_settings()
    engine = GraphitiEngine(settings=settings)
    driver = AsyncGraphDatabase.driver(
        settings.neo4j_uri, auth=(settings.neo4j_username, settings.neo4j_password)
    )

    updated = 0
    try:
        async with driver.session(database=settings.neo4j_database) as session:
            result = await session.run(
                "MATCH ()-[e:RELATES_TO]->() "
                "WHERE e.fact IS NOT NULL "
                "RETURN e.uuid AS uuid, e.fact AS fact "
                "LIMIT $batch_size",
                batch_size=BATCH_SIZE,
            )
            records = [record async for record in result]

        for record in records:
            embedding = await engine.graphiti.embedder.create(input_data=[record["fact"]])
            async with driver.session(database=settings.neo4j_database) as session:
                await session.run(
                    "MATCH ()-[e:RELATES_TO {uuid: $uuid}]->() "
                    "SET e.fact_embedding = $embedding",
                    uuid=record["uuid"],
                    embedding=embedding,
                )
            updated += 1

        logger.info("Re-embedded %d edges using model=%s", updated, new_embedding_model)
        return updated
    finally:
        await driver.close()
        await engine.close()


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    asyncio.run(backfill_stale_embeddings(new_embedding_model="text-embedding-3-large"))
