# ForgeAI Quality Report — MaritimeLogisticsAssistant

## 1. Executive Summary
- **Brief:** I need a maritime logistics and port-operations assistant for vessel planners to look up port codes, coordinates, draft limits, port call costs, and container fleet vessel classes/capacities from the LINERLIB dataset, calculate total port call or voyage cost estimates, and check live port weather conditions when coordinates are available.
- **Initial Pass Rate:** 95.0%
- **Final Pass Rate:** 95.0% (19/20 cases passed)
- **Iterations Run:** 0
- **Budget Used:** 155,928 tokens | 221.0s

## 2. Groundability Guard Findings
- **Tool-Supported Capabilities:**
  - port codes and coordinates via csv_lookup
  - draft limits via csv_lookup
  - port call costs via csv_lookup
  - container fleet vessel classes/capacities via csv_lookup
  - total port call or voyage cost estimates via calculator
  - live port weather conditions via open_meteo_weather
  - Structured tabular lookup via csv_lookup tool
- All remaining requested topics in the brief were groundable in the documents or registered tools.

## 3. Tool Selection Audit
- **`csv_lookup`**: The brief requires looking up port codes, coordinates, draft limits, port call costs, and container fleet vessel classes/capacities from the LINERLIB dataset, which are all supported by the csv_lookup tool.
- **`calculator`**: The brief explicitly asks for calculating total port call or voyage cost estimates, which is supported by the calculator tool.
- **`open_meteo_weather`**: The brief requires checking live port weather conditions when coordinates are available, which is supported by the open_meteo_weather tool.

## 4b. Eval Grounding Validation Audit
- **Non-Refusal Cases Validated:** 15/15
- **Facts Checked / Kept / Dropped:** 33 / 22 / 14
- **Cases Regenerated:** 4

## 5. Critic–Optimizer Iteration Scorecard
- No optimization iterations were required or budget stopped early.

## 6. Honest Failure Analysis (What Still Fails & Why)
- **Remaining Failure Breakdown:** {'wrong_refusal': 1}

- **`ans_08` (answerable → `wrong_refusal`):** What is the fixed port call cost for the port of Tilbury?
  - *Why it failed:* Agent wrongly refused an in-scope, answerable question.
