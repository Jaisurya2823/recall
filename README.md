# Recall

A personal journaling agent with **persistent, semantic memory** — built for the
CockroachDB × AWS "Agentic Memory" hackathon.

You write journal entries. Later, you ask questions in plain English
("what was I stressed about last week?") and the agent retrieves the
semantically relevant past entries from CockroachDB.

## ⚠️ Submission scope note

This submission uses **CockroachDB fully as intended**, but does **not**
integrate a real AWS service — an AWS account could not be created in time
(AWS requires a card for verification, which wasn't available before the
deadline). As a result:

- ✅ CockroachDB Distributed Vector Indexing — fully implemented and working
- ✅ CockroachDB Cloud Managed MCP Server — used during development
- ❌ AWS Bedrock — **not connected**. `bedrock_mock.py` stands in for it,
  using a simple deterministic word-hash embedding instead of a real model,
  so the write → embed → store → retrieve pipeline is still fully
  demonstrable end-to-end without external API calls.

This means the submission does not meet the hackathon's "at least one AWS
service" requirement and may not be eligible for judging/prizes on that
basis — flagging this transparently rather than claiming AWS integration
that isn't there.

## How it works

**CockroachDB (what's real):**
- Every journal entry is embedded and stored in a `VECTOR` column with a
  `vector_cosine_ops` index directly in CockroachDB.
- Semantic search runs as a normal SQL query
  (`ORDER BY embedding <-> $query LIMIT 5`) — no separate vector database.

**Embedding/answering (mocked):**
- `bedrock_mock.py` generates a deterministic pseudo-embedding from word
  hashes (not true semantic understanding) and returns the raw retrieved
  memories instead of an LLM-generated answer.
- Swapping in real AWS Bedrock later just means replacing this one file
  with real Titan Embeddings + Claude calls — the CockroachDB layer
  (`db.py`) doesn't need to change at all.

## Setup

### 1. CockroachDB Cloud
1. Create a free cluster at https://cockroachlabs.cloud
2. Grab the connection string from the "Connect" panel
3. Put it in `.env` as `COCKROACHDB_URL`

### 2. Install & run
\`\`\`bash
python -m venv venv
source venv/bin/activate
pip install -r requirements.txt
cp .env.example .env   # then fill in COCKROACHDB_URL

python app.py init
python app.py write "Studied for DBMS exam, felt stressed about normalization"
python app.py write "Went for a walk, felt calmer after talking to a friend"
python app.py ask "what was I stressed about?"
python app.py clear    # wipes all entries, for re-testing
\`\`\`

## Architecture

\`\`\`
 User (CLI)
    |
    v
 app.py -----> bedrock_mock.py --(word-hash embedding, no external API)-->
    |
    v
  db.py -----> CockroachDB
                 - journal_entries (content, embedding VECTOR, created_at)
                 - vector index (vector_cosine_ops) for semantic search
\`\`\`

## License
MIT — see \`LICENSE\`.