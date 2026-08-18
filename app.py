"""
Recall -- a personal journaling agent with persistent, semantic memory.

Built for the CockroachDB x AWS "Agentic Memory" hackathon.

  - CockroachDB stores every entry AND its embedding (VECTOR column + vector index)
    so memory is always-on, transactional, and survives restarts/regions.
  - AWS Bedrock (Titan Embeddings + Claude) powers the embedding and the
    natural-language recall.

Usage:
    python app.py write "Studied for DBMS exam, felt stressed about normalization"
    python app.py ask "what was I stressed about last week?"
    python app.py list
"""

import sys
from dotenv import load_dotenv
from rich.console import Console
from rich.panel import Panel

import db
import bedrock

load_dotenv()
console = Console()


def cmd_write(text: str):
    console.print("[dim]Embedding entry...[/dim]")
    embedding = bedrock.embed_text(text)
    entry_id = db.save_entry(text, embedding)
    console.print(f"[green]Saved.[/green] (id: {entry_id})")


def cmd_ask(question: str):
    console.print("[dim]Embedding question...[/dim]")
    q_embedding = bedrock.embed_text(question)

    console.print("[dim]Searching memory (CockroachDB vector index)...[/dim]")
    results = db.search_similar(q_embedding, limit=5)
    memories = [r["content"] for r in results]

    console.print("[dim]Asking Claude (Bedrock) with retrieved context...[/dim]")
    answer = bedrock.ask(question, memories)

    console.print(Panel(answer, title="Answer", border_style="cyan"))

    if memories:
        console.print("\n[bold]Retrieved memories:[/bold]")
        for r in results:
            console.print(f"  [dim]{r['created_at']}[/dim] -- {r['content']}")


def cmd_list():
    entries = db.all_entries()
    if not entries:
        console.print("[dim]No entries yet.[/dim]")
        return
    for e in entries:
        console.print(f"[dim]{e['created_at']}[/dim] -- {e['content']}")


def main():
    if len(sys.argv) < 2:
        console.print(__doc__)
        return

    command = sys.argv[1]

    if command == "init":
        db.init_db()
        console.print("[green]Database initialized.[/green]")
    elif command == "write":
        text = " ".join(sys.argv[2:])
        if not text:
            console.print("[red]Usage: python app.py write \"your entry\"[/red]")
            return
        cmd_write(text)
    elif command == "ask":
        question = " ".join(sys.argv[2:])
        if not question:
            console.print("[red]Usage: python app.py ask \"your question\"[/red]")
            return
        cmd_ask(question)
    elif command == "list":
        cmd_list()
    else:
        console.print(f"[red]Unknown command: {command}[/red]")
        console.print(__doc__)


if __name__ == "__main__":
    main()
