# ForgeAI — The Agent That Builds Agents

ForgeAI is an autonomous meta-agent factory that compiles a one-paragraph brief and a folder of documents into a declarative, runnable, and tested AI agent—running every stage on a small, imperfect open model.

## Architecture

1. **Document Profiler & Hybrid Indexer** (`ingestion.py`): Profiles PDFs, CSVs (auto-detecting comma and tab delimiters), TXT, and Markdown into a compact `CorpusManifest` and builds an offline BM25 + TF-IDF Reciprocal Rank Fusion retriever with multi-document diversity.
2. **Intake Interviewer & Tool-Aware Groundability Guard** (`interviewer.py`): Evaluates the brief against both the `CorpusManifest` and the 4 registered tools (`csv_lookup`, `calculator`, `open_meteo_weather`, `current_date`), asks at most 5 clarifying questions (or auto-resolves with `--unattended`), separates `tool_supported_capabilities` from genuine `unsupported_gaps`, and populates `must_refuse_topics`.
3. **Tool Selection Audit & Declarative Agent Generator** (`generator.py`): Runs a dedicated per-tool audit (`select_tools`) and produces `agent_config.json` with modular `SystemPromptSections`, `RetrievalConfig`, `OutputContract`, and justified `ToolSelection`.
4. **Chunk-Grounded Eval Generator & Fact Validator** (`eval_gen.py`): Generates ≥20 test cases (`eval_set.json`) bottom-up across `answerable`, `refusal`, `edge_multi_doc`, and `edge_ambiguous`, validates every non-refusal `expected_fact` strictly against its cited source chunks (`eval_validation.json`), and exports `eval_generation_prompts.json`.
5. **Hybrid Deterministic + Decomposed Judge** (`judge.py`): Combines deterministic checks (retrieval hit, citation presence, refusal detection) with binary per-fact and grounding checks at `temperature=0.0`.
6. **Targeted Critic–Optimizer Loop** (`optimizer.py`): Classifies failures (`retrieval_miss`, `hallucination`, `wrong_refusal`, `prompt_defect`), patches only the targeted component (up to 3 iterations with rollback on regression and token/latency budget enforcement), and writes `iteration_*.json` + `scorecard.json`.
7. **Honest Reporter** (`reporter.py`): Produces `final_report.md` detailing groundability findings, tool selection audit, eval validation stats, per-iteration diffs, and every remaining failure.

## Setup (Reproducible)

### 1. Install Dependencies & Editable Package
```bash
pip install -r requirements.txt
pip install -e . --no-build-isolation --no-deps
```

### 2. Configure Open Model Endpoint
Copy `.env.example` to `.env` and configure your OpenAI-compatible open-model endpoint (local **Ollama** `qwen2.5:7b-instruct` by default, or **Groq** / **OpenRouter**):
```bash
cp .env.example .env
```

### 3. Run Unit & End-to-End Tests
```bash
pytest -v
```

## CLI & UI Usage

Run all commands directly from the repo root (no `PYTHONPATH` required after editable install):

### Build an Agent (Unattended Review Mode)
```bash
python -m forgeai.cli build --brief demos/hr_policy/brief.txt --docs demos/hr_policy/docs --out demos/hr_policy/output --unattended
python -m forgeai.cli build --brief demos/maritime_ports/brief.txt --docs demos/maritime_ports/docs --out demos/maritime_ports/output --unattended
python -m forgeai.cli build --brief demos/unseen_rehearsal/brief.txt --docs demos/unseen_rehearsal/docs --out demos/unseen_rehearsal/output --unattended
python -m forgeai.cli build --brief demos/unseen_rehearsal_2/brief.txt --docs demos/unseen_rehearsal_2/docs --out demos/unseen_rehearsal_2/output --unattended
```

Add `--tournament` to enable the Prompt Tournament stretch goal, or omit `--unattended` for the interactive ≤5-question interview.

### Run Any Generated Agent (Generic Chat Runner)
```bash
python -m forgeai.cli chat --config demos/maritime_ports/output/agent_config.json --docs demos/maritime_ports/docs
```

### Launch Streamlit Web UI
```bash
streamlit run src/forgeai/app.py
```

### Regression Mode (After Documents Change)
```bash
python -m forgeai.cli regress --docs demos/hr_policy/docs --artifacts demos/hr_policy/output
```

## Live End-to-End Run Summary (`qwen2.5:7b-instruct` via Ollama)

| Demo | Domain & Docs | Pass Rate (Init → Final) | Iterations | Selected Tools | Groundability Guard Gaps | Eval Validation |
| :--- | :--- | :---: | :---: | :--- | :--- | :--- |
| **Demo 1** (`demos/hr_policy/`) | U.S. OPM & DOL Leave PDFs (`opm_childbirth_adoption_leave.pdf`, `dol_fmla_employee_guide.pdf`) | **75.0% → 75.0%** (15/20) | 2 | `[]` | `Personal tax advice`, `Salary negotiations`, `Commercial maritime crew contracts` | 15/15 cases, 22/27 kept (5 dropped, 2 regenerated) |
| **Demo 2** (`demos/maritime_ports/`) | LINERLIB TSV/CSV + Docs (`ports.csv`, `fleet_data.csv`, `linerlib_readme.md`) | **95.0% → 95.0%** (19/20) | 0 | `csv_lookup`, `calculator`, `open_meteo_weather` | `None` (`[]`) | 15/15 cases, 22/33 kept (14 dropped, 4 regenerated) |
| **Unseen Rehearsal 1** (`demos/unseen_rehearsal/`) | IT Asset & Warranty Helpdesk (`hardware_inventory.csv`, `it_asset_policy.md`, `support_sla.txt`) | **95.0% → 95.0%** (19/20) | 0 | `csv_lookup`, `calculator`, `current_date` | `Submit Jira purchase approval tickets` | 15/15 cases, 22/28 kept (6 dropped, 1 regenerated) |
| **Unseen Rehearsal 2** (`demos/unseen_rehearsal_2/`) | Coastal Cold-Chain & Reefer Ops (`terminal_tariffs.csv`, `cold_chain_sop.md`, `hazardous_cargo_exceptions.txt`, `inspection_checklist.md`) | **90.0% → 90.0%** (18/20) | 0 | `csv_lookup`, `calculator`, `open_meteo_weather` | `Filing customs clearance manifests` | 15/15 cases, 22/24 kept (2 dropped, 0 regenerated) |
