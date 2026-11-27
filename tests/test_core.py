"""Deterministic unit tests for ForgeAI core modules."""

from __future__ import annotations

from pathlib import Path

from forgeai.eval_gen import _fallback_verbatim_fact
from forgeai.ingestion import HybridRetriever, chunk_corpus, profile_corpus
from forgeai.interviewer import _reconcile_tool_capabilities
from forgeai.judge import _detect_refusal_deterministic
from forgeai.llm import _extract_json_block
from forgeai.schemas import DocumentChunk, RequirementsDoc, UnsupportedGap
from forgeai.tools import execute_tool, format_tool_catalog, get_openai_tools


def test_calculator_tool() -> None:
    assert execute_tool("calculator", {"expression": "(120 * 4) + 20"}) == "500"
    assert "error" in execute_tool("calculator", {"expression": "__import__('os').system('echo hi')"}).lower()


def test_current_date_tool() -> None:
    res = execute_tool("current_date", {})
    assert "local_date" in res and "utc_iso" in res


def test_csv_lookup_tool(tmp_path: Path) -> None:
    csv_file = tmp_path / "spares.csv"
    csv_file.write_text("part_no,name,price_usd\nSP-101,Fuel Filter,45\nSP-202,Oil Pump,320\n", encoding="utf-8")
    res = execute_tool("csv_lookup", {"query": "Fuel Filter"}, docs_dir=str(tmp_path))
    assert "SP-101" in res
    assert "45" in res


def test_get_openai_tools_filters_strictly() -> None:
    tools = get_openai_tools(["calculator"])
    assert len(tools) == 1
    assert tools[0]["function"]["name"] == "calculator"


def test_ingestion_and_hybrid_retriever(tmp_path: Path) -> None:
    (tmp_path / "leave.md").write_text(
        "# Annual Leave Policy\nShore staff receive 24 days of paid annual leave per year.",
        encoding="utf-8",
    )
    (tmp_path / "travel.txt").write_text(
        "Business Travel\nEconomy class flights are approved for trips under 6 hours.",
        encoding="utf-8",
    )
    manifest = profile_corpus(str(tmp_path))
    assert manifest.total_files == 2

    chunks = chunk_corpus(str(tmp_path), chunk_size=300, chunk_overlap=40)
    assert len(chunks) >= 2

    retriever = HybridRetriever(chunks)
    hits = retriever.retrieve("How many days of paid annual leave do shore staff get?", top_k=1)
    assert hits[0].filename == "leave.md"


def test_json_extraction_and_refusal_detection() -> None:
    raw = "Here is the output:\n```json\n{\"a\": 1}\n```"
    assert _extract_json_block(raw) == '{"a": 1}'
    assert _detect_refusal_deterministic("I cannot answer this because it is outside my scope.")
    assert not _detect_refusal_deterministic("Shore staff receive 24 days of annual leave.")


def test_tsv_csv_delimiter_parsing() -> None:
    maritime_docs = Path("demos/maritime_ports/docs")
    manifest = profile_corpus(str(maritime_docs))
    docs_by_name = {d.filename: d for d in manifest.documents}

    assert len(docs_by_name["ports.csv"].csv_columns) == 12
    assert "UNLocode" in docs_by_name["ports.csv"].csv_columns
    assert "Latitude" in docs_by_name["ports.csv"].csv_columns
    assert len(docs_by_name["fleet_data.csv"].csv_columns) == 11
    assert "Vessel class" in docs_by_name["fleet_data.csv"].csv_columns

    lookup_raw = execute_tool("csv_lookup", {"query": "Aberdeen", "filename": "ports.csv"}, docs_dir=str(maritime_docs))
    assert '"UNLocode": "GBABD"' in lookup_raw
    assert '"Country": "United Kingdom"' in lookup_raw


def test_format_tool_catalog() -> None:
    catalog = format_tool_catalog()
    for tool_name in ("csv_lookup", "calculator", "open_meteo_weather", "current_date"):
        assert tool_name in catalog


def test_reconcile_tool_capabilities_dedupes_and_filters() -> None:
    manifest = profile_corpus("demos/maritime_ports/docs")
    req = RequirementsDoc(
        brief="Look up ports and vessel classes, calculate fees, check weather, and submit Jira tickets.",
        agent_name="TestAgent",
        scope="Test scope",
        audience="Planners",
        tone="Concise",
        must_answer_topics=[
            "Look up port codes",
            "Handle requests to submit Jira purchase approval tickets",
        ],
        must_refuse_topics=[
            "Container fleet vessel classes/capacities",
            "Submitting Jira purchase approval tickets",
        ],
        tool_supported_capabilities=[
            "Answering policy questions from PDF",
            "Live weather lookup via open_meteo_weather",
        ],
        unsupported_gaps=[
            UnsupportedGap(
                requested_topic="Container fleet vessel classes/capacities",
                reason="Not found",
            ),
            UnsupportedGap(
                requested_topic="Submitting Jira purchase approval tickets",
                reason="No Jira tool available",
            ),
        ],
    )
    reconciled = _reconcile_tool_capabilities(req.brief, manifest, req)
    assert all("open_meteo_weather" in c or "calculator" in c or "csv_lookup" in c for c in reconciled.tool_supported_capabilities)
    assert not any("Answering policy" in c for c in reconciled.tool_supported_capabilities)
    assert not any("vessel" in g.requested_topic.lower() for g in reconciled.unsupported_gaps)
    assert any("jira" in g.requested_topic.lower() for g in reconciled.unsupported_gaps)
    assert not any("jira" in t.lower() for t in reconciled.must_answer_topics)
    assert "Look up port codes" in reconciled.must_answer_topics


def test_fallback_verbatim_fact_selects_matching_row() -> None:
    chunk = DocumentChunk(
        chunk_id="ports.csv::rows_415_420",
        filename="ports.csv",
        page_or_row="rows 415-420",
        text=(
            "Row 415: UNLocode: FRSNG | name: St Nazaire | Country: France | Longitude: 44.12 | Latitude: 4.36\n"
            "Row 416: UNLocode: TWTXG | name: Taichung | Country: Taiwan | Longitude: 120.51 | Latitude: 24.28\n"
            "Row 417: UNLocode: MXTAM | name: Tampico | Country: Mexico | Longitude: -97.85 | Latitude: 22.25"
        ),
    )
    fact = _fallback_verbatim_fact(
        [chunk],
        "What are the longitude and latitude coordinates for the port of Tampico in Mexico?",
    )
    assert "Tampico" in fact
    assert "MXTAM" in fact
    assert "St Nazaire" not in fact


def test_retriever_three_file_diversity() -> None:
    chunks = chunk_corpus("demos/maritime_ports/docs", chunk_size=700, chunk_overlap=100)
    retriever = HybridRetriever(chunks)
    hits = retriever.retrieve(
        "What is the total cost for a vessel with a container fleet of 1000 FEU to make a port call at Douala, "
        "considering a full container load and transfer costs?",
        top_k=5,
        mode="hybrid",
    )
    assert "ports.csv" in {c.filename for c in hits}
    assert any("douala" in c.text.lower() for c in hits)


def test_csv_lookup_wrong_column_fallback() -> None:
    res = execute_tool(
        "csv_lookup",
        {"query": "Aberdeen", "filename": "ports.csv", "column": "PortCallCostFixed"},
        docs_dir="demos/maritime_ports/docs",
    )
    assert "GBABD" in res
    assert "Aberdeen" in res



def test_extract_cited_sources_normalizes_chunks_and_lists():
    from forgeai.runner import _extract_cited_sources
    from forgeai.schemas import DocumentChunk

    retrieved = [
        DocumentChunk(chunk_id="c1", filename="expense_faq.txt", page_or_row="chunk 1", text="faq"),
        DocumentChunk(chunk_id="c2", filename="travel_policy.md", page_or_row="chunk 2", text="policy"),
    ]
    answer = (
        "Summary [Source: expense_faq.txt | chunk 1, travel_policy.md | chunk 2] "
        "and [Source: expense_faq.txt | chunk 2]."
    )
    assert _extract_cited_sources(answer, retrieved) == ["expense_faq.txt", "travel_policy.md"]



def test_refusal_detector_ignores_policy_not_covered_phrasing():
    # Policy rules saying an expense is "not covered under daily per-diem" are NOT agent refusals
    assert not _detect_refusal_deterministic(
        "Client dinners are not covered under the standard daily meal per-diem; they are billed under Client Entertainment."
    )
    assert _detect_refusal_deterministic(
        "This topic is not covered in the provided documents."
    )


def test_stratified_tournament_slice_covers_all_categories():
    from forgeai.pipeline import _select_stratified_tournament_slice
    from forgeai.schemas import EvalCase, EvalSet

    cases = (
        [EvalCase(id=f"ans_{i}", category="answerable", question=f"Q{i}", expected_facts=["f"]) for i in range(10)]
        + [EvalCase(id=f"ref_{i}", category="refusal", question=f"R{i}", should_refuse=True) for i in range(5)]
        + [EvalCase(id=f"multi_{i}", category="edge_multi_doc", question=f"M{i}", expected_facts=["f"]) for i in range(3)]
        + [EvalCase(id=f"amb_{i}", category="edge_ambiguous", question=f"A{i}", expected_facts=["f"]) for i in range(2)]
    )
    sliced = _select_stratified_tournament_slice(EvalSet(agent_name="Demo", cases=cases), limit=8)
    assert len(sliced.cases) == 8
    cats = {c.category for c in sliced.cases}
    assert cats == {"answerable", "refusal", "edge_multi_doc", "edge_ambiguous"}



def test_calculator_sanitizes_currency_and_guards_exponents():
    assert execute_tool("calculator", {"expression": "$2,500 * 0.67 + $18"}) == "1693"
    err = execute_tool("calculator", {"expression": "9 ** 99"})
    assert "error" in err.lower()


def test_csv_lookup_handles_punctuated_queries():
    maritime_docs = Path("demos/maritime_ports/docs")
    res = execute_tool("csv_lookup", {"query": "Aberdeen, United Kingdom?", "filename": "ports.csv"}, docs_dir=str(maritime_docs))
    assert "GBABD" in res


def test_regression_and_baseline_reports_classify_deltas():
    from forgeai.reporter import build_baseline_comparison_report, build_regression_report
    from forgeai.schemas import EvalCase, EvalSet, JudgeResult, RunBudget, Scorecard

    eval_set = EvalSet(
        agent_name="DemoAgent",
        cases=[
            EvalCase(id="c1", category="answerable", question="Q1?", expected_facts=["f1"]),
            EvalCase(id="c2", category="answerable", question="Q2?", expected_facts=["f2"]),
        ],
    )
    sc_base = Scorecard(
        agent_name="DemoAgent",
        total_cases=2,
        initial_pass_rate=0.5,
        final_pass_rate=0.5,
        passed_cases=1,
        failed_cases=1,
        final_results=[
            JudgeResult(
                case_id="c1", category="answerable", passed=True,
                retrieval_hit=True, citation_present=True, refused=False, refusal_correct=True,
            ),
            JudgeResult(
                case_id="c2", category="answerable", passed=False,
                retrieval_hit=True, citation_present=True, refused=False, refusal_correct=True,
                failure_category="prompt_defect", failure_reason="missed f2",
            ),
        ],
        budget=RunBudget(),
    )
    sc_other = Scorecard(
        agent_name="HumanAgent",
        total_cases=2,
        initial_pass_rate=0.5,
        final_pass_rate=0.5,
        passed_cases=1,
        failed_cases=1,
        final_results=[
            JudgeResult(
                case_id="c1", category="answerable", passed=False,
                retrieval_hit=False, citation_present=True, refused=False, refusal_correct=True,
                failure_category="retrieval_miss", failure_reason="missed doc",
            ),
            JudgeResult(
                case_id="c2", category="answerable", passed=True,
                retrieval_hit=True, citation_present=True, refused=False, refusal_correct=True,
            ),
        ],
        budget=RunBudget(),
    )
    reg_md = build_regression_report(sc_base, sc_other, eval_set)
    assert "Newly Broken Cases (1)" in reg_md and "Newly Passing Cases (1)" in reg_md
    comp_md = build_baseline_comparison_report(sc_base, sc_other, eval_set)
    assert "Where ForgeAI Factory Beats Human Baseline (1)" in comp_md
    assert "Where Human Baseline Beats ForgeAI Factory (1)" in comp_md



def test_input_fingerprint_detects_brief_and_upload_changes():
    import hashlib
    from types import SimpleNamespace

    # Load pure fingerprint logic without executing top-level Streamlit page code
    app_src = Path("src/forgeai/app.py").read_text(encoding="utf-8")
    assert "def _compute_input_fingerprint(" in app_src

    def _mock_upload(name: str, payload: bytes):
        return SimpleNamespace(name=name, getvalue=lambda: payload)

    ns: dict = {"hashlib": hashlib, "Path": Path}
    start = app_src.index("def _docs_signature(")
    end = app_src.index("@st.cache_resource")
    exec(app_src[start:end], ns)
    fp = ns["_compute_input_fingerprint"]

    u1 = [_mock_upload("policy.md", b"v1")]
    u2 = [_mock_upload("policy.md", b"v2")]
    u3 = [_mock_upload("other.md", b"v1")]

    f_base = fp("Upload files", "Brief A", "./output/run", u1, "")
    assert f_base == fp("Upload files", "Brief A ", "./output/run", u1, "")
    assert f_base != fp("Upload files", "Brief B", "./output/run", u1, "")
    assert f_base != fp("Upload files", "Brief A", "./output/run", u2, "")
    assert f_base != fp("Upload files", "Brief A", "./output/run", u3, "")
    assert f_base != fp("Upload files", "Brief A", "./output/other", u1, "")
    assert f_base != fp("Use local paths", "Brief A", "./output/run", [], "./demos/hr_policy/docs")



def test_requirement_coverage_separates_refusal_and_includes_doc_evidence():
    from forgeai.reporter import build_requirement_coverage
    from forgeai.schemas import (
        AgentConfig,
        EvalCase,
        EvalSet,
        OutputContract,
        RetrievalConfig,
        SystemPromptSections,
        ToolSelection,
    )

    req = RequirementsDoc(
        brief="Answer mileage rules, calculate costs, refuse personal tax advice.",
        agent_name="TravelDemo",
        scope="Corporate travel policy",
        audience="Engineers",
        tone="Concise",
        must_answer_topics=["Personal vehicle mileage reimbursement"],
        must_refuse_topics=["Personal tax advice"],
        tool_supported_capabilities=["Cost calculation via calculator"],
        unsupported_gaps=[UnsupportedGap(requested_topic="Booking flights in Concur", reason="No Concur tool")],
    )
    cfg = AgentConfig(
        agent_name="TravelDemo",
        description="Travel helper",
        system_prompt=SystemPromptSections(role="r", instructions="i", guardrails="g", refusal_policy="p"),
        retrieval_config=RetrievalConfig(),
        selected_tools=[ToolSelection(name="calculator", justification="Math")],
        output_contract=OutputContract(),
    )
    eval_set = EvalSet(
        agent_name="TravelDemo",
        cases=[
            EvalCase(
                id="ans_01",
                category="answerable",
                question="What is the personal vehicle mileage reimbursement rate?",
                expected_source_docs=["travel_policy.md"],
                expected_facts=["$0.67 per mile"],
                should_refuse=False,
            ),
            EvalCase(
                id="ref_01",
                category="refusal",
                question="Can you give me personal tax advice for my travel deductions?",
                should_refuse=True,
            ),
        ],
    )
    matrix = build_requirement_coverage(req, cfg, eval_set)
    by_topic = {r.topic: r for r in matrix.requirements}
    assert by_topic["Personal vehicle mileage reimbursement"].status == "covered"
    assert by_topic["Personal vehicle mileage reimbursement"].matching_case_ids == ["ans_01"]
    assert "travel_policy.md" in by_topic["Personal vehicle mileage reimbursement"].evidence_sources
    assert by_topic["Personal tax advice"].status == "covered"
    assert by_topic["Personal tax advice"].matching_case_ids == ["ref_01"]
    assert by_topic["Booking flights in Concur"].status == "untested"
    assert "tool:calculator" in by_topic["Cost calculation via calculator"].evidence_sources



def test_deterministic_external_write_guard_catches_4b_missed_gaps():
    manifest = profile_corpus("demos/maritime_ports/docs")
    brief_with_scada = (
        "Answer SOP questions and look up site capacities in CSV.\n"
        "5. Remotely trip circuit breakers and dispatch SCADA inverter setpoints directly on live substations.\n"
        "It must refuse personal tax advice."
    )
    # Simulate a weak 4B model that missed the SCADA gap completely
    req = RequirementsDoc(
        brief=brief_with_scada,
        agent_name="MicrogridAgent",
        scope="Microgrid ops",
        audience="Engineers",
        tone="Concise",
        must_answer_topics=[
            "Site capacities",
            "Remotely trip circuit breakers and dispatch SCADA inverter setpoints",
        ],
        must_refuse_topics=["Personal tax advice"],
        tool_supported_capabilities=[],
        unsupported_gaps=[],
    )
    reconciled = _reconcile_tool_capabilities(brief_with_scada, manifest, req)
    assert any("scada" in g.requested_topic.lower() for g in reconciled.unsupported_gaps)
    assert all("scada" not in t.lower() for t in reconciled.must_answer_topics)

    # Read-only brief must produce zero false-positive external write gaps
    readonly_brief = "Answer HR leave policy questions from OPM and DOL guides. Refuse tax advice."
    req_ro = RequirementsDoc(
        brief=readonly_brief,
        agent_name="HRAgent",
        scope="HR leave",
        audience="Staff",
        tone="Formal",
        must_answer_topics=["FMLA eligibility"],
        must_refuse_topics=["Tax advice"],
        unsupported_gaps=[],
    )
    reconciled_ro = _reconcile_tool_capabilities(readonly_brief, manifest, req_ro)
    assert reconciled_ro.unsupported_gaps == []



def test_format_chat_markdown_cleans_latex_and_escapes_currency():
    import re as _re

    app_src = Path("src/forgeai/app.py").read_text(encoding="utf-8")
    ns: dict = {"re": _re}
    start = app_src.index("def _format_chat_markdown(")
    end = app_src.index("with tab_factory:")
    exec(app_src[start:end], ns)
    fmt = ns["_format_chat_markdown"]

    raw = r"Revenue at $68/MWh: \(96 \times 68 \text{ USD} = $6,528\)."
    out = fmt(raw)
    assert r"\times" not in out
    assert r"\(" not in out and r"\)" not in out
    assert "×" in out
    assert r"\$68/MWh" in out and r"\$6,528" in out
