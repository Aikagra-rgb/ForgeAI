# ForgeAI — The Agent That Builds Agents

**Live Hosted Web App:** https://forgeai-studio.streamlit.app/ *(Streamlit Community Cloud, powered by small open-weight `nvidia/nemotron-3.5-bolt-e4b` via NVIDIA NIM)*

ForgeAI is an autonomous meta-agent factory that compiles a one-paragraph brief and a folder of documents into a declarative, runnable, and tested AI agent—running every stage on an open-weight model.

## Open-Model Runtime (Provider-Agnostic)

Every stage runs through a single OpenAI-compatible client (`llm.py`) with zero code changes across providers:
- **Local / Offline CLI & Committed Demo Runs:** `qwen2.5:7b-instruct` via local **Ollama** (`http://localhost:11434/v1`) — used to generate all 4 committed demo runs and the Factory vs. Human Baseline comparison offline with zero rate limits.
- **Hosted Web Demo (Streamlit Community Cloud):** `nvidia/nemotron-3.5-bolt-e4b` via **NVIDIA NIM** (`https://integrate.api.nvidia.com/v1`) — fast 4B open-weight model powering https://forgeai-studio.streamlit.app/ (reviewers can also supply a session-only custom endpoint/model in the sidebar).

## Architecture

1. **Document Profiler & Hybrid Indexer** (`ingestion.py`): Profiles PDFs, CSVs (auto-detecting comma and tab delimiters), TXT, and Markdown into a compact `CorpusManifest` and builds an offline BM25 + TF-IDF Reciprocal Rank Fusion retriever with multi-document diversity.
2. **Intake Interviewer & Tool-Aware Groundability Guard** (`interviewer.py`): Evaluates the brief against both the `CorpusManifest` and the 4 registered tools (`csv_lookup`, `calculator`, `open_meteo_weather`, `current_date`), asks at most 5 clarifying questions (or auto-resolves with `--unattended`), separates `tool_supported_capabilities` from genuine `unsupported_gaps`, and populates `must_refuse_topics`.
3. **Tool Selection Audit & Declarative Agent Generator** (`generator.py`): Runs a dedicated per-tool audit (`select_tools`) and produces `agent_config.json` with modular `SystemPromptSections`, `RetrievalConfig`, `OutputContract`, and justified `ToolSelection`.
4. **Chunk-Grounded Eval Generator & Fact Validator** (`eval_gen.py`): Generates ≥20 test cases (`eval_set.json`) bottom-up across `answerable`, `refusal`, `edge_multi_doc`, and `edge_ambiguous`, validates every non-refusal `expected_fact` strictly against its cited source chunks (`eval_validation.json`), and exports `eval_generation_prompts.json`.
5. **Hybrid Deterministic + Decomposed Judge** (`judge.py`): Combines deterministic checks (retrieval hit, citation presence, refusal detection) with binary per-fact and grounding checks at `temperature=0.0`.
6. **Targeted Critic–Optimizer Loop** (`optimizer.py`): Classifies failures (`retrieval_miss`, `hallucination`, `wrong_refusal`, `prompt_defect`), patches only the targeted component (up to 3 iterations with rollback on regression and token/latency budget enforcement), and writes `iteration_*.json` + `scorecard.json`.
7. **Honest Reporter & Requirement Coverage Map** (`reporter.py`): Produces `final_report.md` and `coverage_matrix.json`, linking every `must_answer`, `must_refuse`, `tool_capability`, and `unsupported_gap` item to its eval cases and document/tool evidence (flagging untested requirements honestly), alongside groundability findings, tool selection audit, eval validation stats, per-iteration diffs, and every remaining failure.

## Quick Start for Reviewers (Unseen Brief)

1. Install dependencies & editable package (no `PYTHONPATH` required):
   ```bash
   pip install -r requirements.txt
   pip install -e . --no-build-isolation --no-deps
   cp .env.example .env
   ```
2. Run **unattended CLI build** on any brief + documents folder:
   ```bash
   python -m forgeai.cli build --brief path/to/brief.txt --docs path/to/docs --out ./output/unseen --unattended
   ```
3. Or launch the **Streamlit Web UI** (`streamlit run src/forgeai/app.py`):
   - **Tab 1 — Build Agent:** Upload a brief + source files (isolated into unique per-run upload folders) or use local paths → review/edit the drafted **Human Checkpoint `Requirements JSON`** → toggle **Prompt Tournament** (stratified across answerable, refusal, multi-doc, and ambiguous cases) → watch live percentage & per-case progress.
   - **Tab 2 — Run Agent:** Automatically loads the newly built `agent_config.json` & docs folder (with cached retrieval index), shows enabled tools, and logs normalized `Sources | Tools | Latency` captions per reply.
   - **Tab 3 — Regression:** Re-evaluates against an updated documents folder with live progress, Baseline vs. Regression pass-rate delta metrics, and a failed-cases drift table.

## Setup (Reproducible)

### 1. Install Dependencies & Editable Package
```bash
pip install -r requirements.txt
pip install -e . --no-build-isolation --no-deps
```

### 2. Configure Open Model Endpoint
Copy `.env.example` to `.env` and configure your OpenAI-compatible open-model endpoint (local **Ollama** `qwen2.5:7b-instruct` by default, or **NVIDIA NIM** `nvidia/nemotron-3.5-bolt-e4b` / **Groq** / **OpenRouter**):
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

### Factory vs. Human Baseline Comparison
```bash
python -m forgeai.cli compare --baseline demos/hr_policy/human_baseline_config.json --docs demos/hr_policy/docs --artifacts demos/hr_policy/output
```

## Stretch Goals Implemented

- **Prompt Tournament:** Stratified 8-case tournament slice across 3 candidate system prompts (`--tournament` / Streamlit checkbox), logged in `final_report.md`.
- **Budget Enforcement:** Configurable token/latency `RunBudget` (`FORGEAI_MAX_TOKENS`, `FORGEAI_MAX_LATENCY_SECONDS`) with early stopping on plateau/budget exhaustion and automatic rollback on regression.
- **Human Checkpoint:** Review/edit structured `requirements.json` before generation in both the Streamlit UI (editable JSON checkpoint) and interactive CLI (`requirements_draft.json`).
- **Regression Mode:** Re-evaluates updated documents against frozen `eval_set.json`, persists `regression_scorecard.json` + `regression_report.md`, and displays Baseline vs. Regression delta metrics in Tab 3.
- **Factory vs. Human Baseline:** Hand-written `demos/hr_policy/human_baseline_config.json` compared on the exact same 20-case `eval_set.json` (`demos/hr_policy/output/baseline_comparison.md` + Tab 4):
  - **Result (`demos/hr_policy`):** ForgeAI Factory **75.0% (15/20)** vs. Human Baseline **60.0% (12/20)** (**+15.0%**; Factory won 3 cases — `ans_07`, `ref_01`, `multi_03` — via broader out-of-scope refusal generalization and hybrid `top_k=5` retrieval; 5 shared hard failures documented honestly).
- **Web UI & Live Hosting:** 4-tab Streamlit app (`streamlit_app.py` / `src/forgeai/app.py`) deployed live at **https://forgeai-studio.streamlit.app/** on Streamlit Community Cloud using NVIDIA NIM (`nvidia/nemotron-3.5-bolt-e4b`), featuring upload isolation, interactive clarifying questions + SHA-256 stale-checkpoint invalidation, live progress bars, cached retrieval index, and a session-only LLM endpoint sidebar override.

## Live End-to-End Run Summary (`qwen2.5:7b-instruct` via Ollama)

| Demo | Domain & Docs | Pass Rate (Init → Final) | Iterations | Selected Tools | Groundability Guard Gaps | Eval Validation |
| :--- | :--- | :---: | :---: | :--- | :--- | :--- |
| **Demo 1** (`demos/hr_policy/`) | U.S. OPM & DOL Leave PDFs (`opm_childbirth_adoption_leave.pdf`, `dol_fmla_employee_guide.pdf`) | **75.0% → 75.0%** (15/20) | 2 | `[]` | `Personal tax advice`, `Salary negotiations`, `Commercial maritime crew contracts` | 15/15 cases, 22/27 kept (5 dropped, 2 regenerated) |
| **Demo 2** (`demos/maritime_ports/`) | LINERLIB TSV/CSV + Docs (`ports.csv`, `fleet_data.csv`, `linerlib_readme.md`) | **95.0% → 95.0%** (19/20) | 0 | `csv_lookup`, `calculator`, `open_meteo_weather` | `None` (`[]`) | 15/15 cases, 22/33 kept (14 dropped, 4 regenerated) |
| **Unseen Rehearsal 1** (`demos/unseen_rehearsal/`) | IT Asset & Warranty Helpdesk (`hardware_inventory.csv`, `it_asset_policy.md`, `support_sla.txt`) | **95.0% → 95.0%** (19/20) | 0 | `csv_lookup`, `calculator`, `current_date` | `Submit Jira purchase approval tickets` | 15/15 cases, 22/28 kept (6 dropped, 1 regenerated) |
| **Unseen Rehearsal 2** (`demos/unseen_rehearsal_2/`) | Coastal Cold-Chain & Reefer Ops (`terminal_tariffs.csv`, `cold_chain_sop.md`, `hazardous_cargo_exceptions.txt`, `inspection_checklist.md`) | **90.0% → 90.0%** (18/20) | 0 | `csv_lookup`, `calculator`, `open_meteo_weather` | `Filing customs clearance manifests` | 15/15 cases, 22/24 kept (2 dropped, 0 regenerated) |

> *Artifact note:* All new factory runs also persist `eval_set_raw.json` alongside `eval_set.json` and `eval_validation.json`; the 4 earlier committed demo runs contain `eval_set.json` and `eval_validation.json` (recording every dropped/kept fact) from prior to adding `eval_set_raw.json` export.
