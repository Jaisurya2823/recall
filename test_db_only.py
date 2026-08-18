"""
test_db_only.py -- verify CockroachDB connectivity and the vector schema
WITHOUT needing AWS Bedrock yet. Uses random vectors as stand-ins for
real embeddings.

Run:
    python test_db_only.py

Once AWS Bedrock is set up, go back to using app.py (which uses real
Titan embeddings) instead of this file.
"""

import random
from dotenv import load_dotenv

import db

load_dotenv()


def fake_embedding(dim=db.EMBEDDING_DIM):
    return [random.uniform(-1, 1) for _ in range(dim)]


def main():
    print("Connecting to CockroachDB and creating table/index...")
    db.init_db()
    print("OK: table + vector index ready.\n")

    print("Inserting a test entry...")
    entry_id = db.save_entry("This is a test journal entry.", fake_embedding())
    print(f"OK: inserted entry {entry_id}\n")

    print("Inserting a second test entry...")
    entry_id_2 = db.save_entry("Another test entry about exams.", fake_embedding())
    print(f"OK: inserted entry {entry_id_2}\n")

    print("Listing all entries:")
    for e in db.all_entries():
        print(f"  {e['created_at']} -- {e['content']}")

    print("\nRunning a similarity search (against a random query vector):")
    results = db.search_similar(fake_embedding(), limit=5)
    for r in results:
        print(f"  distance={r['distance']:.4f} -- {r['content']}")

    print("\nAll good -- CockroachDB connection, schema, and vector index are working.")


if __name__ == "__main__":
    main()