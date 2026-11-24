# ForgeAI Quality Report — Coastal Reefer Operations Assistant

## 1. Executive Summary
- **Brief:** I need a coastal cold-chain and reefer operations assistant for terminal dispatchers to look up coastal terminal codes, coordinates, reefer plug-in tariffs, free dwell days, and temperature limits from the terminal tariffs CSV, calculate multi-day dwell and power surcharge costs, check live terminal weather conditions, and answer cold-chain SOP, hazardous cargo, and pre-trip inspection questions from our policy docs. The assistant should also support remotely adjusting reefer temperature setpoints via IoT telemetry and filing customs clearance manifests.
- **Initial Pass Rate:** 90.0%
- **Final Pass Rate:** 90.0% (18/20 cases passed)
- **Iterations Run:** 0
- **Budget Used:** 125,373 tokens | 212.5s

## 2. Groundability Guard Findings
- **Tool-Supported Capabilities:**
  - Lookup terminal codes, coordinates, reefer plug-in tariffs, free dwell days, and temperature limits via terminal_tariffs.csv via csv_lookup
  - Calculate multi-day dwell and power surcharge costs via calculator
  - Check live terminal weather conditions via open_meteo_weather
- **Unsupported:** `Filing customs clearance manifests` — The available tools do not support external write actions or third-party integrations such as filing customs clearance manifests.

## 3. Tool Selection Audit
- **`csv_lookup`**: The brief requires looking up terminal codes, coordinates, reefer plug-in tariffs, free dwell days, and temperature limits from the terminal tariffs CSV.
- **`calculator`**: The brief explicitly asks to calculate multi-day dwell and power surcharge costs.
- **`open_meteo_weather`**: The brief explicitly asks to check live terminal weather conditions.

## 4b. Eval Grounding Validation Audit
- **Non-Refusal Cases Validated:** 15/15
- **Facts Checked / Kept / Dropped:** 24 / 22 / 2
- **Cases Regenerated:** 0

## 5. Critic–Optimizer Iteration Scorecard
- No optimization iterations were required or budget stopped early.

## 6. Honest Failure Analysis (What Still Fails & Why)
- **Remaining Failure Breakdown:** {'prompt_defect': 1, 'wrong_refusal': 1}

- **`ans_09` (answerable → `prompt_defect`):** What is the reefer plug-in charge for a container with a chargeable dwell of 5 days if the daily reefer plug-in rate is $100 and the power surcharge percentage is 20%?
  - *Why it failed:* Missed expected facts (0%): ['total reefer charge equals 500 * (1 + 20 / 100)']
- **`ref_04` (refusal → `wrong_refusal`):** Could you tell me about the procedures for handling hazardous cargo at the terminals?
  - *Why it failed:* Agent answered an out-of-scope/unsupported question instead of refusing.
