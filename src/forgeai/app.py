"""Streamlit UI for building, reviewing, and running ForgeAI agents."""

from __future__ import annotations

import hashlib
import os
import re
import sys
import uuid
from pathlib import Path

_SRC_DIR = str(Path(__file__).resolve().parents[1])
if _SRC_DIR not in sys.path:
    sys.path.insert(0, _SRC_DIR)

import streamlit as st

from forgeai.pipeline import run_baseline_comparison, run_factory, run_regression
from forgeai.ingestion import profile_corpus
from forgeai.interviewer import probe_intake, run_intake_interview, synthesize_requirements
from forgeai.llm import LLMClient
from forgeai.runner import GenericAgentRunner
from forgeai.schemas import AgentConfig, ClarifyingQA, RequirementsDoc, Scorecard

# Bridge Streamlit Cloud secrets (if present) into environment variables for LLMClient
for _secret_key in (
    "FORGEAI_BASE_URL",
    "FORGEAI_API_KEY",
    "FORGEAI_MODEL",
    "FORGEAI_MAX_TOKENS",
    "FORGEAI_MAX_LATENCY_SECONDS",
):
    if _secret_key not in os.environ:
        try:
            if _secret_key in st.secrets:
                os.environ[_secret_key] = str(st.secrets[_secret_key])
        except Exception:
            pass


st.set_page_config(page_title="ForgeAI — The Agent That Builds Agents", layout="wide")
st.title("ForgeAI — The Agent That Builds Agents")

with st.sidebar:
    st.header("LLM Endpoint")
    _env_url = os.getenv("FORGEAI_BASE_URL", "http://localhost:11434/v1")
    _env_model = os.getenv("FORGEAI_MODEL", "qwen2.5:7b-instruct")
    st.caption(f"Default endpoint: `{_env_url}`  \nDefault model: `{_env_model}`")
    with st.expander("Session override (optional)"):
        st.caption("Leave blank to use server `.env` / Streamlit Secrets. Overrides stay in your browser session only.")
        _override_url = st.text_input("Base URL", value="", placeholder="https://integrate.api.nvidia.com/v1", key="llm_override_url").strip()
        _override_model = st.text_input("Model", value="", placeholder="z-ai/glm-5.3-flash", key="llm_override_model").strip()
        _override_key = st.text_input("API Key", value="", type="password", key="llm_override_key").strip()


def _make_llm_client() -> LLMClient:
    """Create an LLMClient honoring optional session-only sidebar overrides."""
    return LLMClient(
        base_url=_override_url or None,
        api_key=_override_key or None,
        model=_override_model or None,
    )


def _has_llm_override() -> bool:
    return bool(_override_url or _override_key or _override_model)


tab_factory, tab_chat, tab_regression, tab_baseline = st.tabs(
    ["1. Build Agent", "2. Run Agent", "3. Regression", "4. Factory vs Baseline"]
)


def _safe_filename(name: str) -> str:
    """Keep uploads as flat, safe filenames while retaining their extensions."""
    leaf = Path(name.replace("\\", "/")).name
    safe = re.sub(r"[^A-Za-z0-9._-]+", "_", leaf).strip("._")
    if not safe:
        raise ValueError("Uploaded file has an invalid filename.")
    return safe


def _docs_signature(docs_dir: str) -> tuple[tuple[str, int], ...]:
    """Compute a lightweight mtime signature for files in docs_dir for cache invalidation."""
    root = Path(docs_dir)
    if not root.exists():
        return ()
    return tuple(
        sorted((p.name, p.stat().st_mtime_ns) for p in root.iterdir() if p.is_file())
    )


def _compute_input_fingerprint(
    input_mode: str,
    brief_text: str,
    output_dir: str,
    uploaded_docs: list | None = None,
    local_docs_dir: str = "",
) -> str:
    """Fingerprint active build inputs so changing brief/files invalidates any pending checkpoint."""
    h = hashlib.sha256()
    h.update(input_mode.encode("utf-8"))
    h.update(b"\x00")
    h.update(brief_text.strip().encode("utf-8"))
    h.update(b"\x00")
    h.update(output_dir.strip().encode("utf-8"))
    h.update(b"\x00")
    if input_mode == "Upload files":
        entries: list[tuple[str, int, str]] = []
        for upload in uploaded_docs or []:
            raw = upload.getvalue()
            entries.append((upload.name, len(raw), hashlib.sha256(raw).hexdigest()))
        for name, size, digest in sorted(entries):
            h.update(f"{name}:{size}:{digest};".encode("utf-8"))
    else:
        resolved = str(Path(local_docs_dir).expanduser().resolve()) if local_docs_dir.strip() else ""
        h.update(resolved.encode("utf-8"))
        for fname, mtime_ns in _docs_signature(resolved):
            h.update(f"{fname}:{mtime_ns};".encode("utf-8"))
    return h.hexdigest()


@st.cache_resource(show_spinner=False)
def _get_cached_runner(
    cfg_path: str,
    cfg_mtime_ns: int,
    docs_dir: str,
    docs_sig: tuple[tuple[str, int], ...],
) -> GenericAgentRunner:
    """Cache GenericAgentRunner and its BM25/TF-IDF index across chat turns."""
    del cfg_mtime_ns, docs_sig
    cfg = AgentConfig.model_validate_json(Path(cfg_path).read_text(encoding="utf-8"))
    return GenericAgentRunner(config=cfg, docs_dir=docs_dir)


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
    col_opt1, col_opt2 = st.columns(2)
    tournament = col_opt1.checkbox("Run Prompt Tournament", value=False)
    interactive_intake = col_opt2.checkbox("Ask clarifying questions first (≤5)", value=False)

    if "factory_finished" not in st.session_state:
        st.session_state.factory_finished = None
    if "pending_requirements" not in st.session_state:
        st.session_state.pending_requirements = None
    if "pending_probe" not in st.session_state:
        st.session_state.pending_probe = None
    if "pending_manifest" not in st.session_state:
        st.session_state.pending_manifest = None
    if "pending_input_fingerprint" not in st.session_state:
        st.session_state.pending_input_fingerprint = None
    if "build_inputs" not in st.session_state:
        st.session_state.build_inputs = None

    current_fingerprint = _compute_input_fingerprint(
        input_mode=input_mode,
        brief_text=brief_text,
        output_dir=output_dir,
        uploaded_docs=uploaded_docs,
        local_docs_dir=local_docs_dir,
    )
    if (
        st.session_state.pending_input_fingerprint is not None
        and st.session_state.pending_input_fingerprint != current_fingerprint
    ):
        st.session_state.pending_requirements = None
        st.session_state.pending_probe = None
        st.session_state.pending_manifest = None
        st.session_state.build_inputs = None
        st.session_state.analyze_summary = None
        st.session_state.pending_input_fingerprint = None
        st.session_state.stale_draft_notice = True

    if st.session_state.get("stale_draft_notice"):
        st.warning(
            "Inputs changed since the last analysis — cleared the stale draft. "
            "Click **Analyze brief and prepare requirements** to analyze the current brief and documents."
        )

    if st.button("Analyze brief and prepare requirements", type="primary"):
        st.session_state.stale_draft_notice = False
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
            analyze_bar = st.progress(0.15, text="15% — Profiling documents...")
            manifest = profile_corpus(str(docs_dir))
            doc_names = [d.filename for d in manifest.documents]
            total_words = sum(d.word_count for d in manifest.documents)
            st.session_state.build_inputs = {
                "brief": brief_text.strip(),
                "docs_dir": str(docs_dir),
                "output_dir": str(out),
                "input_mode": input_mode,
                "doc_names": doc_names,
            }
            st.session_state.pending_input_fingerprint = current_fingerprint
            st.session_state.factory_finished = None

            if interactive_intake:
                analyze_bar.progress(
                    0.50,
                    text=f"50% — Profiled {len(manifest.documents)} files ({total_words:,} words). Generating clarifying questions...",
                )
                probe = probe_intake(brief=brief_text.strip(), manifest=manifest, llm=_make_llm_client())
                analyze_bar.progress(1.0, text="100% — Clarifying questions ready!")
                st.session_state.pending_probe = probe
                st.session_state.pending_manifest = manifest
                st.session_state.pending_requirements = None
                st.session_state.analyze_summary = None
            else:
                analyze_bar.progress(
                    0.45,
                    text=f"45% — Profiled {len(manifest.documents)} files ({total_words:,} words). Drafting requirements...",
                )
                requirements = run_intake_interview(
                    brief=brief_text.strip(), manifest=manifest, llm=_make_llm_client(), unattended=True
                )
                analyze_bar.progress(1.0, text="100% — Requirements ready for review!")
                req_json = requirements.model_dump_json(indent=2)
                st.session_state.pending_probe = None
                st.session_state.pending_manifest = None
                st.session_state.pending_requirements = req_json
                st.session_state.requirements_checkpoint = req_json
                st.session_state.analyze_summary = {
                    "files": doc_names,
                    "total_words": total_words,
                    "tool_caps": requirements.tool_supported_capabilities,
                    "gaps": [g.requested_topic for g in requirements.unsupported_gaps],
                }
        except Exception as exc:
            st.error(f"Build failed: {exc}")

    if st.session_state.pending_probe is not None and st.session_state.pending_manifest is not None:
        probe = st.session_state.pending_probe
        manifest = st.session_state.pending_manifest
        inputs = st.session_state.build_inputs
        total_words = sum(d.word_count for d in manifest.documents)
        st.subheader("Interactive Intake — Clarifying Questions")
        st.caption("Answer any clarifying questions below (leave blank to use grounded defaults), then draft the requirements JSON.")
        st.info(
            f"**Brief:** {inputs['brief']}  \n"
            f"**Documents ({len(inputs['doc_names'])}):** {', '.join(inputs['doc_names'])} · {total_words:,} words  \n"
            f"**Initial gaps detected:** {[g.requested_topic for g in probe.unsupported_gaps] or ['none']}"
        )
        answers: list[str] = []
        for idx, q in enumerate(probe.clarifying_questions[:5], start=1):
            ans = st.text_input(f"Q{idx}. {q}", key=f"intake_q_{idx}")
            answers.append(ans.strip())
        if st.button("Draft Requirements JSON from Answers", type="primary", key="synthesize_req_btn"):
            try:
                if st.session_state.pending_input_fingerprint != current_fingerprint:
                    raise ValueError("Inputs changed — please re-run analysis.")
                qa_pairs = [
                    ClarifyingQA(
                        question=q,
                        answer=a or "Use conservative, document- and tool-grounded defaults.",
                    )
                    for q, a in zip(probe.clarifying_questions[:5], answers)
                ]
                with st.spinner("Synthesizing structured requirements..."):
                    requirements = synthesize_requirements(
                        brief=inputs["brief"],
                        manifest=manifest,
                        probe=probe,
                        qa_pairs=qa_pairs,
                        llm=_make_llm_client(),
                    )
                req_json = requirements.model_dump_json(indent=2)
                st.session_state.pending_requirements = req_json
                st.session_state.requirements_checkpoint = req_json
                st.session_state.analyze_summary = {
                    "files": inputs["doc_names"],
                    "total_words": total_words,
                    "tool_caps": requirements.tool_supported_capabilities,
                    "gaps": [g.requested_topic for g in requirements.unsupported_gaps],
                }
                st.session_state.pending_probe = None
                st.session_state.pending_manifest = None
                st.rerun()
            except Exception as exc:
                st.error(f"Could not draft requirements: {exc}")

    if st.session_state.pending_requirements:
        st.subheader("Human checkpoint")
        st.caption("Review or edit the structured requirements. The factory continues only after approval.")
        inputs = st.session_state.build_inputs or {}
        summary = st.session_state.get("analyze_summary")
        if summary:
            st.info(
                f"**Brief being approved:** {inputs.get('brief', '')}  \n"
                f"**Documents ({len(summary['files'])}):** {', '.join(summary['files'])} · "
                f"{summary['total_words']:,} words  \n"
                f"**Output folder:** `{inputs.get('output_dir', '')}`  \n"
                f"**Tool capabilities detected:** {summary['tool_caps'] or ['none']}  \n"
                f"**Unsupported gaps flagged:** {summary['gaps'] or ['none']}"
            )
        edited_requirements = st.text_area(
            "Requirements JSON",
            height=420, key="requirements_checkpoint",
        )
        if st.button("Approve requirements and build agent", type="primary", key="approve_requirements"):
            try:
                if st.session_state.pending_input_fingerprint != current_fingerprint:
                    raise ValueError("Inputs changed since analysis. Re-analyze before building.")
                approved = RequirementsDoc.model_validate_json(edited_requirements)
                inputs = st.session_state.build_inputs
                progress_bar = st.progress(0.0, text="Starting build...")
                status_caption = st.empty()

                def _update_build_progress(pct: float, msg: str) -> None:
                    clamped = max(0.0, min(1.0, pct))
                    progress_bar.progress(clamped, text=f"{int(clamped * 100)}% — {msg}")
                    status_caption.caption(msg)

                cfg, scorecard, report_path = run_factory(
                    brief=inputs["brief"], docs_dir=inputs["docs_dir"], output_dir=inputs["output_dir"],
                    unattended=True, tournament=tournament, requirements_override=approved,
                    llm=_make_llm_client(),
                    progress_bar_fn=_update_build_progress,
                )
                built_cfg_path = str(Path(inputs["output_dir"]) / "agent_config.json")
                st.session_state.factory_finished = {
                    "config_path": built_cfg_path,
                    "docs_dir": inputs["docs_dir"], "report_path": str(report_path),
                    "agent_name": cfg.agent_name, "pass_rate": scorecard.final_pass_rate,
                    "passed": scorecard.passed_cases, "total": scorecard.total_cases,
                }
                st.session_state.chat_cfg_path = built_cfg_path
                st.session_state.chat_docs_dir = inputs["docs_dir"]
                st.session_state.regression_docs = inputs["docs_dir"]
                st.session_state.regression_artifacts = inputs["output_dir"]
                st.session_state.messages = []
                st.session_state.pending_requirements = None
                st.session_state.pending_probe = None
                st.session_state.pending_manifest = None
                st.session_state.pending_input_fingerprint = None
                st.rerun()
            except Exception as exc:
                st.error(f"Could not complete build: {exc}")

    result = st.session_state.factory_finished
    if result:
        st.success(
            f"Built {result['agent_name']} — {result['pass_rate']:.1%} "
            f"({result['passed']}/{result['total']} cases passed)"
        )
        st.caption(
            f"Loaded into Tab 2 (Run Agent): `{result['config_path']}` | Docs: `{result['docs_dir']}`"
        )
        st.markdown(Path(result["report_path"]).read_text(encoding="utf-8"))

with tab_chat:
    st.subheader("Generic Agent Chat Runner")
    if "chat_cfg_path" not in st.session_state:
        st.session_state.chat_cfg_path = "./demos/hr_policy/output/agent_config.json"
    if "chat_docs_dir" not in st.session_state:
        st.session_state.chat_docs_dir = "./demos/hr_policy/docs"
    cfg_path = st.text_input("agent_config.json", key="chat_cfg_path")
    chat_docs_dir = st.text_input("Documents folder for retrieval", key="chat_docs_dir")
    try:
        preview_cfg = AgentConfig.model_validate_json(Path(cfg_path).read_text(encoding="utf-8"))
        enabled_tools = [t.name for t in preview_cfg.selected_tools] or ["none"]
        st.caption(f"Loaded agent: **{preview_cfg.agent_name}** | Enabled tools: `{enabled_tools}`")
    except Exception:
        st.caption("Enter a valid `agent_config.json` path to inspect the loaded agent.")
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
            cfg_file = Path(cfg_path)
            if _has_llm_override():
                cfg = AgentConfig.model_validate_json(cfg_file.read_text(encoding="utf-8"))
                runner = GenericAgentRunner(config=cfg, docs_dir=chat_docs_dir, llm=_make_llm_client())
            else:
                runner = _get_cached_runner(
                    cfg_path=str(cfg_file.resolve()),
                    cfg_mtime_ns=cfg_file.stat().st_mtime_ns,
                    docs_dir=str(Path(chat_docs_dir).resolve()),
                    docs_sig=_docs_signature(chat_docs_dir),
                )
            history = [
                {"role": m["role"], "content": m["content"]}
                for m in st.session_state.messages
            ]
            trace = runner.run(prompt, history=history)
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
    if "regression_docs" not in st.session_state:
        st.session_state.regression_docs = "./demos/hr_policy/docs"
    if "regression_artifacts" not in st.session_state:
        st.session_state.regression_artifacts = "./demos/hr_policy/output"
    regression_docs = st.text_input("Updated documents folder", key="regression_docs")
    regression_artifacts = st.text_input("Existing artifacts folder", key="regression_artifacts")
    if st.button("Run Regression"):
        try:
            reg_progress = st.progress(0.0, text="Starting regression evaluation...")
            reg_caption = st.empty()

            def _update_reg_progress(pct: float, msg: str) -> None:
                clamped = max(0.0, min(1.0, pct))
                reg_progress.progress(clamped, text=f"{int(clamped * 100)}% — {msg}")
                reg_caption.caption(msg)

            scorecard = run_regression(
                docs_dir=regression_docs,
                artifacts_dir=regression_artifacts,
                llm=_make_llm_client(),
                progress_bar_fn=_update_reg_progress,
            )
            baseline_path = Path(regression_artifacts) / "scorecard.json"
            baseline_rate = None
            if baseline_path.exists():
                try:
                    baseline_sc = Scorecard.model_validate_json(baseline_path.read_text(encoding="utf-8"))
                    baseline_rate = baseline_sc.final_pass_rate
                except Exception:
                    baseline_rate = None

            c1, c2, c3, c4 = st.columns(4)
            if baseline_rate is not None:
                delta_pct = (scorecard.final_pass_rate - baseline_rate) * 100.0
                c1.metric("Baseline Pass Rate", f"{baseline_rate:.1%}")
                c2.metric("Regression Pass Rate", f"{scorecard.final_pass_rate:.1%}", delta=f"{delta_pct:+.1f}%")
            else:
                c1.metric("Baseline Pass Rate", "N/A")
                c2.metric("Regression Pass Rate", f"{scorecard.final_pass_rate:.1%}")
            c3.metric("Passed", f"{scorecard.passed_cases}/{scorecard.total_cases}")
            c4.metric("Failed", f"{scorecard.failed_cases}/{scorecard.total_cases}")

            failed_rows = [
                {
                    "Case ID": r.case_id,
                    "Category": r.category,
                    "Failure Type": r.failure_category or "unknown",
                    "Reason": r.failure_reason,
                }
                for r in scorecard.final_results
                if not r.passed
            ]
            if failed_rows:
                st.subheader(f"Detected Regressions / Failures ({len(failed_rows)})")
                st.dataframe(failed_rows, use_container_width=True, hide_index=True)
            else:
                st.success("All saved evaluation cases still pass on the updated documents!")

            with st.expander("Raw Regression Scorecard JSON"):
                st.json(scorecard.model_dump(mode="json"))
        except Exception as exc:
            st.error(f"Regression failed: {exc}")


with tab_baseline:
    st.subheader("Factory vs. Human Baseline")
    st.caption(
        "Compare a factory-generated agent against a hand-written baseline on the exact same saved eval_set.json."
    )
    baseline_cfg_path = st.text_input(
        "Hand-written baseline config",
        value="./demos/hr_policy/human_baseline_config.json",
        key="baseline_cfg_path",
    )
    baseline_docs_dir = st.text_input(
        "Documents folder",
        value="./demos/hr_policy/docs",
        key="baseline_docs_dir",
    )
    baseline_artifacts_dir = st.text_input(
        "Factory artifacts folder",
        value="./demos/hr_policy/output",
        key="baseline_artifacts_dir",
    )
    existing_comp = Path(baseline_artifacts_dir) / "baseline_comparison.md"
    if st.button("Run Factory vs. Human Baseline Comparison", type="primary", key="run_baseline_compare"):
        try:
            comp_bar = st.progress(0.0, text="Starting baseline comparison...")
            comp_caption = st.empty()

            def _update_comp_progress(pct: float, msg: str) -> None:
                clamped = max(0.0, min(1.0, pct))
                comp_bar.progress(clamped, text=f"{int(clamped * 100)}% — {msg}")
                comp_caption.caption(msg)

            fac_sc, hum_sc, comp_report_path = run_baseline_comparison(
                baseline_config_path=baseline_cfg_path,
                docs_dir=baseline_docs_dir,
                artifacts_dir=baseline_artifacts_dir,
                llm=_make_llm_client(),
                progress_bar_fn=_update_comp_progress,
            )
            delta_pct = (fac_sc.final_pass_rate - hum_sc.final_pass_rate) * 100.0
            b1, b2, b3 = st.columns(3)
            b1.metric("Factory Pass Rate", f"{fac_sc.final_pass_rate:.1%}", delta=f"{delta_pct:+.1f}% vs Human")
            b2.metric("Human Baseline Pass Rate", f"{hum_sc.final_pass_rate:.1%}")
            b3.metric("Eval Cases", f"{fac_sc.total_cases}")
            st.markdown(comp_report_path.read_text(encoding="utf-8"))
        except Exception as exc:
            st.error(f"Baseline comparison failed: {exc}")
    elif existing_comp.exists():
        st.info(f"Showing saved comparison report from `{existing_comp}`.")
        st.markdown(existing_comp.read_text(encoding="utf-8"))
