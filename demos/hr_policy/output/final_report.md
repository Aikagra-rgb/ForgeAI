# ForgeAI Quality Report — HRLeaveAssistant

## 1. Executive Summary
- **Brief:** I need an assistant that answers HR leave and workplace flexibility questions based on official US federal/DOL leave guides—covering FMLA eligibility, qualifying reasons, medical certification, and paid parental/adoption leave flexibilities. It must refuse requests for personal tax advice, salary negotiations, or commercial maritime crew contracts not covered in these guides.
- **Initial Pass Rate:** 75.0%
- **Final Pass Rate:** 75.0% (15/20 cases passed)
- **Iterations Run:** 2
- **Budget Used:** 216,744 tokens | 465.3s

## 2. Groundability Guard Findings
- **Unsupported:** `Personal tax advice` — The provided documents and tools do not cover tax advice.
- **Unsupported:** `Salary negotiations` — The provided documents and tools do not cover salary negotiations.
- **Unsupported:** `Commercial maritime crew contracts` — The provided documents and tools do not cover commercial maritime crew contracts.

## 3. Tool Selection Audit
- **No tools attached** (`[]`) — pure document-grounded QA.

## 4b. Eval Grounding Validation Audit
- **Non-Refusal Cases Validated:** 15/15
- **Facts Checked / Kept / Dropped:** 27 / 22 / 5
- **Cases Regenerated:** 2

## 5. Critic–Optimizer Iteration Scorecard
| Iteration | Score Before | Score After | Dominant Failures | Changes Applied |
| :--- | :---: | :---: | :--- | :--- |
| 1 | 75.0% (15/20) | 75.0% (15/20) | {'prompt_defect': 4, 'wrong_refusal': 1} | `system_prompt.instructions` (Clarified the scope and added a directive to avoid external information.); `rollback` (Rolled back patch because pass rate dropped (75.0% -> 70.0%).) |
| 2 | 75.0% (15/20) | 75.0% (15/20) | {'prompt_defect': 4, 'wrong_refusal': 1} | `system_prompt.refusal_policy` (Corrected the refusal policy to ensure clear and firm refusal for out-of-scope requests without attempting partial answers or clarification.); `rollback` (Rolled back patch because pass rate dropped (75.0% -> 70.0%).) |

## 6. Honest Failure Analysis (What Still Fails & Why)
- **Remaining Failure Breakdown:** {'prompt_defect': 4, 'wrong_refusal': 1}

- **`ans_06` (answerable → `prompt_defect`):** Under what condition would an employee not be eligible for FMLA leave according to the document?
  - *Why it failed:* Missed expected facts (0%): ['If an employee has 50 or more employees within 75 miles of the worksite, they are not eligible for FMLA leave.']
- **`ans_09` (answerable → `prompt_defect`):** What is the time frame for an employee to use FMLA/PPL for adoption of a fostered child?
  - *Why it failed:* Missed expected facts (0%): ['an employee is pursuing adoption of a child the employee is fostering, the employee may invoke and use FMLA/PPL in the 12-month period following the new placement of the child with the employee for foster care purposes']
- **`ans_10` (answerable → `prompt_defect`):** What is the minimum number of hours an airline flight attendant must work for an employer to be eligible for FMLA leave?
  - *Why it failed:* Missed expected facts (0%): ['The employer must have the employee work at least 1250 hours in the 12 months before taking leave.']
- **`ref_05` (refusal → `wrong_refusal`):** Can you provide information on the process for requesting leave under the Family and Medical Leave Act (FMLA)?
  - *Why it failed:* Agent answered an out-of-scope/unsupported question instead of refusing.
- **`multi_02` (edge_multi_doc → `prompt_defect`):** What must an employer do if an employee is taking leave related to the birth of a child under FMLA conditions?
  - *Why it failed:* Missed expected facts (0%): ['The employer must notify the employee if the leave is designated as FMLA leave.', 'The employer must inform the employee how much leave will be counted against the FMLA entitlement if possible.', 'If the leave is not covered by FMLA, the employer must notify the employee of that determination.']
