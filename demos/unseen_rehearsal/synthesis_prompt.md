# Synthesis Prompt — Unseen Rehearsal: IT Asset & Warranty Helpdesk

Synthesize a fresh unseen domain dataset for an unattended live rehearsal:
1. `docs/hardware_inventory.csv` — 8 IT hardware assets (`asset_id,model,category,department,unit_price_usd,stock_qty,warranty_expiry`) with warranty dates spanning past (`2025`, early `2026`) and future (`2027`, `2028`) dates so `current_date` warranty-expiry checks are testable.
2. `docs/it_asset_policy.md` — Corporate hardware refresh, bulk discount tiers (5–9 units: 10% discount; 10+ units: 15% discount), out-of-warranty replacement guidelines, and explicit note that purchase approvals require manual submission in Jira outside this helpdesk.
3. `docs/support_sla.txt` — IT support response SLAs by severity (P1: 30 minutes, P2: 4 hours, P3: 1 business day) and temporary loaner hardware rules.
