-- Mock document: relational reporting mirror for the Email Knowledge Graph.
-- Neo4j holds the live Graphiti graph; this schema is a periodic flattened
-- export used by the BI dashboard team, who query it with plain SQL.

CREATE TABLE email_threads (
    thread_id       VARCHAR(64) PRIMARY KEY,
    subject         VARCHAR(512) NOT NULL,
    started_at      TIMESTAMP NOT NULL,
    last_message_at TIMESTAMP NOT NULL,
    message_count   INTEGER NOT NULL DEFAULT 0
);

CREATE TABLE emails (
    message_id   VARCHAR(64) PRIMARY KEY,
    thread_id    VARCHAR(64) NOT NULL REFERENCES email_threads(thread_id),
    sender       VARCHAR(256) NOT NULL,
    recipient    VARCHAR(256),
    subject      VARCHAR(512) NOT NULL,
    sent_at      TIMESTAMP NOT NULL,
    body_text    TEXT
);

CREATE TABLE graph_entities (
    entity_uuid  VARCHAR(64) PRIMARY KEY,
    entity_name  VARCHAR(256) NOT NULL,
    entity_type  VARCHAR(64),
    group_id     VARCHAR(128) NOT NULL,
    created_at   TIMESTAMP NOT NULL
);

CREATE TABLE graph_facts (
    edge_uuid         VARCHAR(64) PRIMARY KEY,
    source_entity_uuid VARCHAR(64) NOT NULL REFERENCES graph_entities(entity_uuid),
    target_entity_uuid VARCHAR(64) NOT NULL REFERENCES graph_entities(entity_uuid),
    fact              TEXT NOT NULL,
    valid_at          TIMESTAMP,
    invalid_at        TIMESTAMP,
    group_id          VARCHAR(128) NOT NULL
);

-- Facts that are currently active (not superseded by a newer episode).
CREATE VIEW current_facts AS
SELECT * FROM graph_facts WHERE invalid_at IS NULL;

-- Example: find the current meeting time discussed with a given person.
-- SELECT fact FROM current_facts f
-- JOIN graph_entities e ON f.source_entity_uuid = e.entity_uuid
-- WHERE e.entity_name = 'Jane Doe' AND f.fact ILIKE '%meeting%';
