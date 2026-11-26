# ForgeAI Factory vs. Human Baseline — HRLeaveAssistant

## 1. Head-to-Head Summary
| Metric | ForgeAI Factory | Human Baseline | Delta (Factory - Human) |
| :--- | :---: | :---: | :---: |
| **Pass Rate** | **75.0%** (15/20) | **60.0%** (12/20) | **+15.0%** |
| **Both Pass / Both Fail** | 12 passed both | 5 failed both | Factory wins: 3 · Human wins: 0 |

## 2. Where ForgeAI Factory Beats Human Baseline (3)
- **`ans_07` (answerable):** What activities can an employee engage in while taking FMLA leave related to the birth of a child?
  - *Why human baseline failed:* `wrong_refusal` — Agent wrongly refused an in-scope, answerable question.
- **`ref_01` (refusal):** Is there any guidance on how to handle leave requests during a company-wide emergency situation?
  - *Why human baseline failed:* `wrong_refusal` — Agent answered an out-of-scope/unsupported question instead of refusing.
- **`multi_03` (edge_multi_doc):** What information should Theo have regarding his FMLA leave usage and adoption process, based on the provided documents?
  - *Why human baseline failed:* `prompt_defect` — Missed expected facts (0%): ['FMLA with substitution of PPL is only available for a new placement with an employee for adoption or foster care.']

## 3. Where Human Baseline Beats ForgeAI Factory (0)
- None on this eval set.

## 4. Shared Failure Modes (5)
- **`ans_06` (answerable):** Under what condition would an employee not be eligible for FMLA leave according to the document?
  - *Factory:* `prompt_defect` (Missed expected facts (0%): ['If an employee has 50 or more employees within 75 miles of the worksite, they are not eligible for FMLA leave.'])
  - *Human:* `prompt_defect` (Missed expected facts (0%): ['If an employee has 50 or more employees within 75 miles of the worksite, they are not eligible for FMLA leave.'])
- **`ans_09` (answerable):** What is the time frame for an employee to use FMLA/PPL for adoption of a fostered child?
  - *Factory:* `prompt_defect` (Missed expected facts (0%): ['an employee is pursuing adoption of a child the employee is fostering, the employee may invoke and use FMLA/PPL in the 12-month period following the new placement of the child with the employee for foster care purposes'])
  - *Human:* `prompt_defect` (Missed expected facts (0%): ['an employee is pursuing adoption of a child the employee is fostering, the employee may invoke and use FMLA/PPL in the 12-month period following the new placement of the child with the employee for foster care purposes'])
- **`ans_10` (answerable):** What is the minimum number of hours an airline flight attendant must work for an employer to be eligible for FMLA leave?
  - *Factory:* `prompt_defect` (Missed expected facts (0%): ['The employer must have the employee work at least 1250 hours in the 12 months before taking leave.'])
  - *Human:* `prompt_defect` (Missed expected facts (0%): ['The employer must have the employee work at least 1250 hours in the 12 months before taking leave.'])
- **`ref_05` (refusal):** Can you provide information on the process for requesting leave under the Family and Medical Leave Act (FMLA)?
  - *Factory:* `wrong_refusal` (Agent answered an out-of-scope/unsupported question instead of refusing.)
  - *Human:* `wrong_refusal` (Agent answered an out-of-scope/unsupported question instead of refusing.)
- **`multi_02` (edge_multi_doc):** What must an employer do if an employee is taking leave related to the birth of a child under FMLA conditions?
  - *Factory:* `prompt_defect` (Missed expected facts (0%): ['The employer must notify the employee if the leave is designated as FMLA leave.', 'The employer must inform the employee how much leave will be counted against the FMLA entitlement if possible.', 'If the leave is not covered by FMLA, the employer must notify the employee of that determination.'])
  - *Human:* `prompt_defect` (Missed expected facts (0%): ['The employer must notify the employee if the leave is designated as FMLA leave.', 'The employer must inform the employee how much leave will be counted against the FMLA entitlement if possible.', 'If the leave is not covered by FMLA, the employer must notify the employee of that determination.'])
