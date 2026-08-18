"""
bedrock_mock.py -- drop-in stand-in for bedrock.py that needs NO AWS credentials.

Used only for local testing before AWS Bedrock is set up:
  - embed_text() returns a deterministic pseudo-embedding based on word
    overlap, so similar sentences end up numerically closer (rough, but
    good enough to sanity-check the CockroachDB retrieval pipeline).
  - ask() just echoes back the retrieved memories instead of calling an LLM.

Once AWS is ready, app.py should import the real `bedrock` module instead
of this one -- see the one-line swap noted in app.py.
"""

import hashlib

from db import EMBEDDING_DIM


def embed_text(text: str) -> list[float]:
    """Deterministic fake embedding: hash each word into the vector so
    entries sharing words land closer together than unrelated ones."""
    vec = [0.0] * EMBEDDING_DIM
    words = text.lower().split()
    for word in words:
        h = int(hashlib.md5(word.encode()).hexdigest(), 16)
        idx = h % EMBEDDING_DIM
        vec[idx] += 1.0
    # normalize so magnitude doesn't blow up distance comparisons
    norm = sum(v * v for v in vec) ** 0.5
    if norm > 0:
        vec = [v / norm for v in vec]
    else:
        vec = [0.0001] * EMBEDDING_DIM
    return vec


def ask(question: str, memories: list[str]) -> str:
    if not memories:
        return "[MOCK MODE -- no AWS Bedrock] No relevant memories found for that question."
    joined = "\n".join(f"  - {m}" for m in memories)
    return (
        "[MOCK MODE -- no AWS Bedrock, so this is not a real AI answer]\n"
        f"Here are the closest matching memories for \"{question}\":\n{joined}"
    )
