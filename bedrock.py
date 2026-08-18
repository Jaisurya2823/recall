"""
AWS Bedrock layer for Recall.

Two responsibilities:
  1. embed_text()   -> Amazon Titan Embeddings, used to vectorize journal
                        entries before they're stored in CockroachDB.
  2. ask()          -> Claude on Bedrock, used to answer the user's question
                        using retrieved memories as context (RAG).
"""

import os
import json
import boto3

_client = None


def _bedrock():
    global _client
    if _client is None:
        _client = boto3.client(
            "bedrock-runtime",
            region_name=os.environ.get("AWS_REGION", "us-east-1"),
        )
    return _client


def embed_text(text: str) -> list[float]:
    model_id = os.environ.get("BEDROCK_EMBEDDING_MODEL_ID", "amazon.titan-embed-text-v2:0")
    body = json.dumps({"inputText": text})
    response = _bedrock().invoke_model(
        modelId=model_id,
        body=body,
        contentType="application/json",
        accept="application/json",
    )
    result = json.loads(response["body"].read())
    return result["embedding"]


def ask(question: str, memories: list[str]) -> str:
    model_id = os.environ.get("BEDROCK_CHAT_MODEL_ID", "anthropic.claude-3-5-sonnet-20241022-v2:0")

    if memories:
        context = "\n".join(f"- {m}" for m in memories)
        system_prompt = (
            "You are the user's personal memory journal assistant. "
            "Use the retrieved past journal entries below to answer their "
            "question accurately. If the entries don't contain the answer, "
            "say so honestly instead of guessing.\n\n"
            f"Retrieved memories:\n{context}"
        )
    else:
        system_prompt = (
            "You are the user's personal memory journal assistant. "
            "No relevant past entries were found for this question -- say so."
        )

    body = json.dumps({
        "anthropic_version": "bedrock-2023-05-31",
        "max_tokens": 500,
        "system": system_prompt,
        "messages": [{"role": "user", "content": question}],
    })

    response = _bedrock().invoke_model(
        modelId=model_id,
        body=body,
        contentType="application/json",
        accept="application/json",
    )
    result = json.loads(response["body"].read())
    return result["content"][0]["text"]
