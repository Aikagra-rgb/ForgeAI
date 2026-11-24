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
