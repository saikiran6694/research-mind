"""
ResearchMind — main entry point.

Usage:
  python main.py "your research topic"
  python main.py "your topic" --depth deep
  python main.py "your topic" --depth shallow --focus "ethics" "policy"
  python main.py --list-checkpoints
"""

import sys
import json
import argparse
from dotenv import load_dotenv
from rich.console import Console
from rich.markdown import Markdown
from rich.panel import Panel
from rich.table import Table
from rich.progress import Progress, SpinnerColumn, TextColumn

load_dotenv()

from graph.builder import build_research_graph
from graph.state import ResearchState
from models.schemas import ResearchQuery, ResearchDepth
from utils.checkpoint import load_checkpoint, clear_checkpoint, list_checkpoints
from utils.rate_limiter import get_llm

console = Console()


def build_initial_state(query: ResearchQuery) -> ResearchState:
    return {
        "query":             query,
        "messages":          [],
        "iteration":         0,
        "phase":             "planning",
        "search_queries":    [],
        "raw_sources":       [],
        "validated_sources": [],
        "findings":          [],
        "knowledge_gaps":    [],
        "critique_notes":    [],
        "report":            None,
        "error":             None,
    }


def restore_state(checkpoint: dict, query: ResearchQuery) -> ResearchState:
    """
    Fix 4: Rebuild a runnable state from a saved checkpoint.
    Pydantic objects are reconstructed; everything else is reused as-is.
    """
    state = build_initial_state(query)
    skip  = {"_checkpoint_phase", "_checkpoint_saved_at", "_checkpoint_topic", "query"}
    for k, v in checkpoint.items():
        if k in skip:
            continue
        if k in state:
            state[k] = v
    return state


def run_research(topic: str, depth: str = "medium",
                 focus_areas: list[str] | None = None):

    query = ResearchQuery(
        topic=topic,
        depth=ResearchDepth(depth),
        focus_areas=focus_areas or [],
    )

    # Fix 4: resume from checkpoint if one exists
    checkpoint = load_checkpoint(topic)
    if checkpoint:
        resume_phase = checkpoint.get("_checkpoint_phase", "unknown")
        console.print(f"[yellow]↩ Resuming from checkpoint at phase: {resume_phase}[/]")
        initial_state = restore_state(checkpoint, query)
    else:
        initial_state = build_initial_state(query)

    graph = build_research_graph()

    console.print(Panel(
        f"[bold cyan]ResearchMind[/] — [yellow]{topic}[/]\n"
        f"Depth: [green]{depth}[/]  |  "
        f"LLM calls today: [magenta]{get_llm().daily_calls}[/]"
    ))

    # LLM call budget display
    call_map = {"shallow": "~3", "medium": "~5", "deep": "~7"}
    console.print(
        f"[dim]Expected LLM calls this run: {call_map.get(depth, '~5')} "
        f"(free-tier optimised)[/]\n"
    )

    final_state = None
    with Progress(
        SpinnerColumn(),
        TextColumn("[progress.description]{task.description}"),
        console=console,
    ) as progress:
        task = progress.add_task("Starting...", total=None)

        for event in graph.stream(initial_state):
            node_name  = list(event.keys())[0]
            node_state = list(event.values())[0]
            phase      = node_state.get("phase", "")
            iteration  = node_state.get("iteration", 0)
            n_sources  = len(node_state.get("raw_sources", []))
            n_findings = len(node_state.get("findings", []))

            progress.update(
                task,
                description=(
                    f"[cyan]{node_name}[/] → phase: {phase} | "
                    f"iter: {iteration} | sources: {n_sources} | "
                    f"findings: {n_findings} | "
                    f"llm calls: {get_llm().daily_calls}"
                ),
            )
            final_state = node_state

    _print_report(final_state, topic)


def _print_report(state: dict, topic: str):
    if not state:
        console.print("[red]No state returned.[/]")
        return

    report = state.get("report")
    if not report:
        console.print("[red]Research did not produce a report.[/]")
        console.print(f"Error: {state.get('error')}")
        return

    console.print("\n")
    console.print(Panel("[bold green]✓ Research Complete[/]"))

    # Summary
    console.print(Markdown(f"## {report.topic}\n\n{report.summary}"))

    # Stats table
    table = Table(show_header=False, box=None, padding=(0, 2))
    table.add_row("Sources used",    str(len(report.sources)))
    table.add_row("Key findings",    str(len(report.key_findings)))
    table.add_row("Overall confidence", f"{report.confidence_overall:.0%}")
    table.add_row("Word count",      str(report.word_count))
    table.add_row("LLM calls used",  str(get_llm().daily_calls))
    console.print(table)

    # Gaps
    if report.gaps_identified:
        console.print("\n[bold]Knowledge gaps identified:[/]")
        for g in report.gaps_identified:
            console.print(f"  • {g}")

    # Follow-up
    if report.follow_up_queries:
        console.print("\n[bold]Suggested follow-up queries:[/]")
        for q in report.follow_up_queries:
            console.print(f"  → {q}")

    # Save to JSON
    slug     = topic[:30].replace(" ", "_")
    out_path = f"report_{slug}.json"
    with open(out_path, "w") as f:
        json.dump(report.model_dump(), f, indent=2, default=str)
    console.print(f"\n[dim]Full report saved to {out_path}[/]")

    # Fix 4: clear checkpoint on successful completion
    clear_checkpoint(topic)


def main():
    parser = argparse.ArgumentParser(description="ResearchMind — free-tier optimised")
    parser.add_argument("topic", nargs="*", help="Research topic")
    parser.add_argument("--depth", choices=["shallow", "medium", "deep"],
                        default="medium")
    parser.add_argument("--focus", nargs="*", default=[],
                        help="Focus areas e.g. --focus ethics policy")
    parser.add_argument("--list-checkpoints", action="store_true",
                        help="List all saved checkpoints")
    args = parser.parse_args()

    if args.list_checkpoints:
        cps = list_checkpoints()
        if not cps:
            console.print("No checkpoints found.")
        for cp in cps:
            console.print(
                f"[cyan]{cp['topic']}[/] — phase: {cp['phase']} — saved: {cp['saved']}"
            )
        return

    if not args.topic:
        console.print("[red]Please provide a research topic.[/]")
        console.print("Usage: python main.py \"your topic\" --depth medium")
        sys.exit(1)

    topic = " ".join(args.topic)
    run_research(topic, depth=args.depth, focus_areas=args.focus)


if __name__ == "__main__":
    main()
