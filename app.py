"""
ResearchMind — Streamlit UI.

Usage:
  streamlit run app.py
"""

import json
import time
import streamlit as st
from dotenv import load_dotenv

load_dotenv()

from graph.builder import build_research_graph
from models.schemas import ResearchQuery, ResearchDepth
from utils.checkpoint import load_checkpoint, clear_checkpoint, list_checkpoints
from utils.rate_limiter import get_llm
from main import build_initial_state, restore_state

st.set_page_config(page_title="ResearchMind", page_icon="🔎", layout="wide")

CALL_ESTIMATE = {"shallow": "~3", "medium": "~5", "deep": "~7"}


def run_research(topic: str, depth: str, focus_areas: list[str]):
    query = ResearchQuery(
        topic=topic,
        depth=ResearchDepth(depth),
        focus_areas=focus_areas,
    )

    checkpoint = load_checkpoint(topic)
    if checkpoint:
        resume_phase = checkpoint.get("_checkpoint_phase", "unknown")
        st.info(f"↩ Resuming from checkpoint at phase: **{resume_phase}**")
        initial_state = restore_state(checkpoint, query)
    else:
        initial_state = build_initial_state(query)

    graph = build_research_graph()

    status_box = st.empty()
    final_state = None

    with st.spinner("Researching..."):
        for event in graph.stream(initial_state):
            node_name = list(event.keys())[0]
            node_state = list(event.values())[0]
            phase = node_state.get("phase", "")
            iteration = node_state.get("iteration", 0)
            n_sources = len(node_state.get("raw_sources", []))
            n_findings = len(node_state.get("findings", []))

            status_box.markdown(
                f"**{node_name}** → phase: `{phase}` | iter: {iteration} | "
                f"sources: {n_sources} | findings: {n_findings} | "
                f"LLM calls today: {get_llm().daily_calls}"
            )
            final_state = node_state

    return final_state


def _stream_words(text: str, delay: float = 0.02):
    for word in text.split(" "):
        yield word + " "
        time.sleep(delay)


def render_report(state: dict, topic: str):
    if not state:
        st.error("No state returned.")
        return

    report = state.get("report")
    if not report:
        st.error("Research did not produce a report.")
        st.write(f"Error: {state.get('error')}")
        return

    st.success("✓ Research complete")
    st.header(report.topic)
    st.write_stream(_stream_words(report.summary))

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Sources used", len(report.sources))
    c2.metric("Key findings", len(report.key_findings))
    c3.metric("Confidence", f"{report.confidence_overall:.0%}")
    c4.metric("Word count", report.word_count)

    st.subheader("Key findings")
    for f in report.key_findings:
        with st.expander(f"{f.claim}  (confidence: {f.confidence:.0%})"):
            for e in f.evidence:
                st.markdown(f"- {e}")
            if f.contradictions:
                st.markdown("**Contradictions:**")
                for c in f.contradictions:
                    st.markdown(f"- {c}")
            if f.source_urls:
                st.markdown("**Sources:** " + ", ".join(f.source_urls))

    if report.gaps_identified:
        st.subheader("Knowledge gaps identified")
        for g in report.gaps_identified:
            st.markdown(f"- {g}")

    if report.follow_up_queries:
        st.subheader("Suggested follow-up queries")
        for q in report.follow_up_queries:
            st.markdown(f"- {q}")

    st.subheader("Sources")
    for s in report.sources:
        st.markdown(f"- [{s.title}]({s.url}) — credibility {s.credibility_score:.0%}")

    report_json = json.dumps(report.model_dump(), indent=2, default=str)
    slug = topic[:30].replace(" ", "_")
    st.download_button(
        "Download full report (JSON)",
        data=report_json,
        file_name=f"report_{slug}.json",
        mime="application/json",
    )

    clear_checkpoint(topic)


def main():
    st.title("🔎 ResearchMind")
    st.caption("Autonomous research agent — plan, search, analyze, critique, synthesize.")

    with st.sidebar:
        st.subheader("New research run")
        topic = st.text_input("Research topic", placeholder="e.g. impact of AI on education")
        depth = st.selectbox("Depth", ["shallow", "medium", "deep"], index=1)
        focus_raw = st.text_input("Focus areas (comma-separated, optional)", placeholder="ethics, policy")
        focus_areas = [f.strip() for f in focus_raw.split(",") if f.strip()]

        st.caption(f"Expected LLM calls this run: {CALL_ESTIMATE.get(depth, '~5')}")
        st.caption(f"LLM calls today: {get_llm().daily_calls}")

        run_clicked = st.button("Start research", type="primary", disabled=not topic)

        st.divider()
        st.subheader("Checkpoints")
        cps = list_checkpoints()
        if not cps:
            st.caption("No saved checkpoints.")
        for cp in cps:
            st.caption(f"**{cp['topic']}** — phase: {cp['phase']}\n\nsaved: {cp['saved']}")

    if run_clicked and topic:
        final_state = run_research(topic, depth, focus_areas)
        render_report(final_state, topic)


if __name__ == "__main__":
    main()
