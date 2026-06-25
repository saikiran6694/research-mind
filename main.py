import json
from dotenv import load_dotenv
from rich.console import Console
from rich.markdown import Markdown
from rich.panel import Panel
from rich.progress import Progress, SpinnerColumn, TextColumn

from graph.builder import build_research_graph
from models.schemas import ResearchQuery


load_dotenv()
console = Console()


def run_research(topic: str, depth: str = "medium", focus_areas: list[str] = None):
    query = ResearchQuery(
        topic=topic,
        depth=depth,
        focus_areas=focus_areas or []
    )

    initial_state = {
        "query": query,
        "messages": [],
        "iteration": 0,
        "phase": "planning",
        "search_queries": [],
        "raw_sources": [],
        "validated_sources": [],
        "findings": [],
        "knowledge_gaps": [],
        "critique_notes": [],
        "report": None,
        "error": None
    }

    graph = build_research_graph()

    console.print(Panel(f"[bold cyan]ResearchMind[/] — Researching: [yellow]{topic}[/]"))

    with Progress(SpinnerColumn(), TextColumn("[progress.description]{task.description}"), console=console) as progress:
        task = progress.add_task("Running research pipeline...", total=None)

        for event in graph.stream(initial_state):
            node_name = list(event.keys())[0]
            state = list(event.values())[0]
            phase = state.get("phase", "")
            progress.update(task, description=f"[cyan]{node_name}[/] → phase: {phase}")
        
    report = state.get("report")
    if report:
        console.print("\n")
        console.print(Panel("[bold green]Research Complete[/]"))
        console.print(Markdown(f"#{report.topic}\n\n{report.summary}"))
        console.print(f"\n**Sources:** {len(report.sources)} | **Findings:** {len(report.key_findings)} | **Confidence:** {report.confidence_overall:.0%}")

        # Save report
        with open(f"report_{topic[:30].replace(' ', '_')}.json", "w") as f:
            json.dump(report.model_dump(), f, indent=2)
        console.print("[dim]Report saved to JSON.[/]")
    else:
        console.print("[red]Research failed — check logs.[/]")



if __name__ == "__main__":
    import sys
    topic = " ".join(sys.argv[1:]) or "impact of large language models on scientific research"

    print(f"Topic: {topic}")

    run_research(topic, depth="medium", focus_areas=["methodology", "real-world applications"])
