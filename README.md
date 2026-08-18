# Recall

A personal journaling agent with **persistent, semantic memory** — built for the
CockroachDB × AWS "Agentic Memory" hackathon.

You write journal entries. Later, you ask questions in plain English
("what was I stressed about last week?") and the agent retrieves the
semantically relevant past entries and answers using them — not keyword
search, actual meaning-based recall.

## How it uses the required tools

**CockroachDB (2 tools used):**
- **Distributed Vector Indexing** — every journal entry is embedded (via AWS
  Bedrock Titan) and stored in a `VECTOR` column with a `vector_cosine_ops`
  index directly in CockroachDB. Semantic search runs as a normal SQL query
  (`ORDER BY embedding <-> $query LIMIT 5`) — no separate vector database,
  no sync gap between the source data and the embeddings.
- **CockroachDB Cloud Managed MCP Server** — used during development to let
  Claude Code inspect the `journal_entries` table, debug the vector index,
  and iterate on schema directly from the terminal.

**AWS (1 service used):**
- **Amazon Bedrock** — Titan Embeddings (`amazon.titan-embed-text-v2:0`)
  generates the vector for each entry and each question; Claude on Bedrock
  (`anthropic.claude-3-5-sonnet-20241022-v2:0`) generates the final answer
  using retrieved memories as context (RAG).

## Setup

### 1. CockroachDB Cloud
1. Create a free cluster at https://cockroachlabs.cloud
2. Grab the connection string from the "Connect" panel
3. Put it in `.env` as `COCKROACHDB_URL`

### 2. AWS Bedrock
1. In the AWS Console, go to Bedrock → Model access, and request access to:
   - `amazon.titan-embed-text-v2:0`
   - `anthropic.claude-3-5-sonnet-20241022-v2:0`
   (Approval is usually instant for Titan, may take a few minutes for Claude.)
2. Create an IAM user/access key with `bedrock:InvokeModel` permission
3. Put the credentials in `.env`

### 3. Install & run
```bash
python -m venv venv
source venv/bin/activate
pip install -r requirements.txt
cp .env.example .env   # then fill in your credentials

python app.py init                     # creates the table + vector index
python app.py write "Studied for DBMS exam, felt stressed about normalization"
python app.py write "Went for a walk, felt much calmer after talking to a friend"
python app.py ask "what was I stressed about recently?"
```

## Architecture

```
 User (CLI)
    |
    v
 app.py -----> bedrock.py --(Titan Embeddings)--> AWS Bedrock
    |                --(Claude RAG answer)------> AWS Bedrock
    |
    v
  db.py -----> CockroachDB
                 - journal_entries (content, embedding VECTOR, created_at)
                 - vector index (vector_cosine_ops) for semantic search
```

## License
MIT — see `LICENSE`.
