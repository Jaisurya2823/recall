"""
CockroachDB layer for Recall.

Uses CockroachDB's native VECTOR type + vector index (available in recent
CockroachDB versions) to store embeddings alongside the journal entries
themselves, so semantic recall lives in the same transactional store as
the source-of-truth data -- no separate vector DB to keep in sync.
"""

import os
import psycopg2
from psycopg2.extras import RealDictCursor

EMBEDDING_DIM = 1024  # amazon.titan-embed-text-v2:0 output size


def get_connection():
    url = os.environ["COCKROACHDB_URL"]
    return psycopg2.connect(url)


def init_db():
    """Create the entries table + vector index if they don't exist yet."""
    conn = get_connection()
    with conn, conn.cursor() as cur:
        cur.execute(f"""
            CREATE TABLE IF NOT EXISTS journal_entries (
                id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
                content TEXT NOT NULL,
                embedding VECTOR({EMBEDDING_DIM}) NOT NULL,
                created_at TIMESTAMPTZ NOT NULL DEFAULT now()
            );
        """)
        # CockroachDB vector index (C_SPC_COSINE = cosine distance)
        cur.execute("""
            CREATE VECTOR INDEX IF NOT EXISTS journal_entries_embedding_idx
            ON journal_entries (embedding vector_cosine_ops);
        """)
    conn.close()


def _vector_literal(embedding: list[float]) -> str:
    """CockroachDB's VECTOR type expects a string literal like '[0.1,0.2,...]',
    not a raw Python list/array -- psycopg2 has no native adapter for it."""
    return "[" + ",".join(repr(float(x)) for x in embedding) + "]"


def save_entry(content: str, embedding: list[float]) -> str:
    conn = get_connection()
    with conn, conn.cursor() as cur:
        cur.execute(
            """
            INSERT INTO journal_entries (content, embedding)
            VALUES (%s, %s::VECTOR)
            RETURNING id;
            """,
            (content, _vector_literal(embedding)),
        )
        entry_id = cur.fetchone()[0]
    conn.close()
    return str(entry_id)


def search_similar(embedding: list[float], limit: int = 5):
    """Return the most semantically similar past entries, nearest first."""
    vec = _vector_literal(embedding)
    conn = get_connection()
    with conn, conn.cursor(cursor_factory=RealDictCursor) as cur:
        cur.execute(
            """
            SELECT id, content, created_at,
                   embedding <-> %s::VECTOR AS distance
            FROM journal_entries
            ORDER BY embedding <-> %s::VECTOR
            LIMIT %s;
            """,
            (vec, vec, limit),
        )
        rows = cur.fetchall()
    conn.close()
    return rows


def all_entries():
    conn = get_connection()
    with conn, conn.cursor(cursor_factory=RealDictCursor) as cur:
        cur.execute("SELECT id, content, created_at FROM journal_entries ORDER BY created_at DESC;")
        rows = cur.fetchall()
    conn.close()
    return rows