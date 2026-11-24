"""Streamlit UI for building, reviewing, and running ForgeAI agents."""

from __future__ import annotations

import re
import uuid
from pathlib import Path

import streamlit as st

from forgeai.pipeline import run_factory, run_regression
from forgeai.ingestion import profile_corpus
from forgeai.interviewer import run_intake_interview
from forgeai.llm import LLMClient
from forgeai.runner import GenericAgentRunner
from forgeai.schemas import AgentConfig, RequirementsDoc


st.set_page_config(page_title="ForgeAI — The Agent That Builds Agents", layout="wide")
st.title("ForgeAI — The Agent That Builds Agents")

tab_factory, tab_chat, tab_regression = st.tabs(
    ["1. Build Agent", "2. Run Agent", "3. Regression"]
)


def _safe_filename(name: str) -> str:
    """Keep uploads as flat, safe filenames while retaining their extensions."""
    leaf = Path(name.replace("\\", "/")).name
    safe = re.sub(r"[^A-Za-z0-9._-]+", "_", leaf).strip("._")
    if not safe:
        raise ValueError("Uploaded file has an invalid filename.")
    return safe


with tab_factory:
    st.subheader("Build from a brief and document folder")
    st.caption("Upload a new brief and its source documents, or use an existing local folder.")
    input_mode = st.radio("Input source", ["Upload files", "Use local paths"], horizontal=True)

    brief_text = ""
    uploaded_docs = []
    local_docs_dir = ""
    if input_mode == "Upload files":
        brief_upload = st.file_uploader("Brief (.txt or .md)", type=["txt", "md"], key="brief_upload")
        if brief_upload is not None:
            brief_text = brief_upload.getvalue().decode("utf-8", errors="replace").strip()
        uploaded_docs = st.file_uploader(
            "Knowledge documents (.pdf, .csv, .txt, .md)",
            type=["pdf", "csv", "txt", "md"],
            accept_multiple_files=True,
            key="docs_upload",
        )
    else:
        brief_text = st.text_area(
            "Brief", value="I need an assistant that answers HR policy questions for shore staff.", height=110
        )
        local_docs_dir = st.text_input("Documents folder", value="./demos/hr_policy/docs")

    output_dir = st.text_input("Output artifacts folder", value="./output/review_run")
    tournament = st.checkbox("Run Prompt Tournament", value=False)

    if "factory_finished" not in st.session_state:
        st.session_state.factory_finished = None
    if "pending_requirements" not in st.session_state:
        st.session_state.pending_requirements = None
    if "build_inputs" not in st.session_state:
        st.session_state.build_inputs = None

    if st.button("Analyze brief and prepare requirements", type="primary"):
        try:
            out = Path(output_dir).expanduser().resolve()
            if input_mode == "Upload files":
                if not brief_text:
                    raise ValueError("Upload a non-empty brief first.")
                if not uploaded_docs:
                    raise ValueError("Upload at least one source document.")
                docs_dir = out / "input_documents" / uuid.uuid4().hex
                docs_dir.mkdir(parents=True, exist_ok=True)
                seen: set[str] = set()
                for upload in uploaded_docs:
                    filename = _safe_filename(upload.name)
                    if filename in seen:
                        raise ValueError(f"Duplicate uploaded filename: {filename}")
                    seen.add(filename)
                    (docs_dir / filename).write_bytes(upload.getvalue())
                (out / "input_brief.txt").write_text(brief_text, encoding="utf-8")
            else:
                docs_dir = Path(local_docs_dir).expanduser().resolve()
                if not brief_text.strip():
                    raise ValueError("Enter a brief.")
            with st.spinner("Profiling documents and drafting requirements..."):
                manifest = profile_corpus(str(docs_dir))
                requirements = run_intake_interview(
                    brief=brief_text.strip(), manifest=manifest, llm=LLMClient(), unattended=True
                )
            req_json = requirements.model_dump_json(indent=2)
            st.session_state.pending_requirements = req_json
            st.session_state.requirements_checkpoint = req_json
            st.session_state.build_inputs = {
                "brief": brief_text.strip(), "docs_dir": str(docs_dir), "output_dir": str(out),
            }
            st.session_state.factory_finished = None
        except Exception as exc:
            st.error(f"Build failed: {exc}")

    if st.session_state.pending_requirements:
        st.subheader("Human checkpoint")
        st.caption("Review or edit the structured requirements. The factory continues only after approval.")
        edited_requirements = st.text_area(
            "Requirements JSON",
            height=420, key="requirements_checkpoint",
        )
        if st.button("Approve requirements and build agent", type="primary", key="approve_requirements"):
            try:
                approved = RequirementsDoc.model_validate_json(edited_requirements)
                inputs = st.session_state.build_inputs
                with st.spinner("Generating, evaluating, and optimizing the agent..."):
                    cfg, scorecard, report_path = run_factory(
                        brief=inputs["brief"], docs_dir=inputs["docs_dir"], output_dir=inputs["output_dir"],
                        unattended=True, tournament=tournament, requirements_override=approved,
                    )
                st.session_state.factory_finished = {
                    "config_path": str(Path(inputs["output_dir"]) / "agent_config.json"),
                    "docs_dir": inputs["docs_dir"], "report_path": str(report_path),
                    "agent_name": cfg.agent_name, "pass_rate": scorecard.final_pass_rate,
                    "passed": scorecard.passed_cases, "total": scorecard.total_cases,
                }
                st.session_state.pending_requirements = None
                st.rerun()
            except Exception as exc:
                st.error(f"Could not complete build: {exc}")

    result = st.session_state.factory_finished
    if result:
        st.success(
            f"Built {result['agent_name']} — {result['pass_rate']:.1%} "
            f"({result['passed']}/{result['total']} cases passed)"
        )
        st.markdown(Path(result["report_path"]).read_text(encoding="utf-8"))

with tab_chat:
    st.subheader("Generic Agent Chat Runner")
    cfg_path = st.text_input("agent_config.json", value="./demos/hr_policy/output/agent_config.json")
    chat_docs_dir = st.text_input("Documents folder for retrieval", value="./demos/hr_policy/docs")
    chat_target = (cfg_path.strip(), chat_docs_dir.strip())
    if st.session_state.get("chat_target") != chat_target:
        st.session_state.chat_target = chat_target
        st.session_state.messages = []
    if "messages" not in st.session_state:
        st.session_state.messages = []
    if st.session_state.messages and st.button("Clear chat", key="clear_chat"):
        st.session_state.messages = []
        st.rerun()
    for msg in st.session_state.messages:
        with st.chat_message(msg["role"]):
            st.markdown(msg["content"])
            if msg.get("meta"):
                st.caption(msg["meta"])
    if prompt := st.chat_input("Ask the generated agent a question..."):
        try:
            cfg = AgentConfig.model_validate_json(Path(cfg_path).read_text(encoding="utf-8"))
            history = [
                {"role": m["role"], "content": m["content"]}
                for m in st.session_state.messages
            ]
            trace = GenericAgentRunner(config=cfg, docs_dir=chat_docs_dir).run(prompt, history=history)
            tools_used = [tc.tool_name for tc in trace.tool_calls]
            meta = (
                f"Sources: {trace.cited_sources or ['none']} | "
                f"Tools: {tools_used or ['none']} | "
                f"Latency: {trace.latency_ms:.0f} ms"
            )
            st.session_state.messages.extend(
                [
                    {"role": "user", "content": prompt},
                    {"role": "assistant", "content": trace.answer, "meta": meta},
                ]
            )
            st.rerun()
        except Exception as exc:
            st.error(f"Could not run agent: {exc}")

with tab_regression:
    st.subheader("Regression evaluation")
    st.caption("Rebuild an agent from the selected requirements and compare it against the saved eval set.")
    regression_docs = st.text_input("Updated documents folder", value="./demos/hr_policy/docs")
    regression_artifacts = st.text_input("Existing artifacts folder", value="./demos/hr_policy/output")
    if st.button("Run Regression"):
        try:
            with st.spinner("Running regression evaluation..."):
                scorecard = run_regression(docs_dir=regression_docs, artifacts_dir=regression_artifacts)
            st.success(
                f"Regression result: {scorecard.final_pass_rate:.1%} "
                f"({scorecard.passed_cases}/{scorecard.total_cases} passed)"
            )
            st.json(scorecard.model_dump(mode="json"))
        except Exception as exc:
            st.error(f"Regression failed: {exc}")
