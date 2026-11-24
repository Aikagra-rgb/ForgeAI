"""Streamlit Web UI for ForgeAI Factory & Generic Chat Runner."""

from __future__ import annotations

from pathlib import Path

import streamlit as st

from forgeai.pipeline import run_factory
from forgeai.runner import GenericAgentRunner
from forgeai.schemas import AgentConfig

st.set_page_config(page_title="ForgeAI — The Agent That Builds Agents", layout="wide")
st.title("ForgeAI — The Agent That Builds Agents")

tab_factory, tab_chat = st.tabs(["1. Build Agent (Factory)", "2. Run Agent (Chat)"])

with tab_factory:
    st.subheader("Compile Brief + Documents into a Tested Agent")
    brief = st.text_area(
        "One-paragraph brief",
        value="I need an assistant that answers HR policy questions for shore staff.",
        height=100,
    )
    col1, col2 = st.columns(2)
    with col1:
        docs_dir = st.text_input("Documents folder path", value="./demos/hr_policy/docs")
    with col2:
        output_dir = st.text_input("Output artifacts folder", value="./demos/hr_policy/output")

    tournament = st.checkbox("Enable Prompt Tournament (stretch goal)", value=False)
    if st.button("Run ForgeAI Factory (Unattended)", type="primary"):
        status_box = st.empty()
        logs: list[str] = []

        def _progress(msg: str) -> None:
            logs.append(msg)
            status_box.info("\n".join(logs))

        with st.spinner("Running ForgeAI pipeline..."):
            cfg, scorecard, report_path = run_factory(
                brief=brief,
                docs_dir=docs_dir,
                output_dir=output_dir,
                unattended=True,
                tournament=tournament,
                progress_fn=_progress,
            )
        st.success(
            f"Built {cfg.agent_name} — Final Pass Rate: {scorecard.final_pass_rate:.1%} "
            f"({scorecard.passed_cases}/{scorecard.total_cases})"
        )
        st.markdown(Path(report_path).read_text(encoding="utf-8"))

with tab_chat:
    st.subheader("Generic Agent Chat Runner")
    cfg_path = st.text_input("Path to agent_config.json", value="./demos/hr_policy/output/agent_config.json")
    chat_docs_dir = st.text_input("Docs folder for retrieval", value="./demos/hr_policy/docs")

    if "messages" not in st.session_state:
        st.session_state.messages = []

    for msg in st.session_state.messages:
        with st.chat_message(msg["role"]):
            st.markdown(msg["content"])

    if prompt := st.chat_input("Ask the generated agent a question..."):
        st.session_state.messages.append({"role": "user", "content": prompt})
        with st.chat_message("user"):
            st.markdown(prompt)

        cfg = AgentConfig.model_validate_json(Path(cfg_path).read_text(encoding="utf-8"))
        runner = GenericAgentRunner(config=cfg, docs_dir=chat_docs_dir)
        trace = runner.run(prompt)

        with st.chat_message("assistant"):
            st.markdown(trace.answer)
            st.caption(f"Cited: {trace.cited_sources} | Tools: {[t.tool_name for t in trace.tool_calls]} | {trace.latency_ms:.0f}ms")
        st.session_state.messages.append({"role": "assistant", "content": trace.answer})
