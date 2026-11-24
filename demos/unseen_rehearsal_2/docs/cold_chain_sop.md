# Coastal Cold-Chain & Reefer Standard Operating Procedure (SOP)

## 1. Cargo Temperature Bands & Excursion Thresholds
- **Pharmaceuticals (Class P):** Maintain strictly between **2°C and 8°C**. Any deviation exceeding ±1.5°C for more than 20 minutes triggers a Critical Excursion Alert.
- **Frozen Seafood (Class F):** Maintain at or below **-18°C**. Temperature rising above -15°C for more than 45 minutes requires immediate reefer unit inspection.
- **Chilled Produce (Class C):** Maintain between **0°C and 4°C** with relative humidity set to 85–90%.
- **Dry Cargo Exceptions:** Non-refrigerated dry containers stored in reefer stacks do not consume shore power and incur zero reefer plug-in charges.

## 2. Dwell Fee & Power Surcharge Billing Formula
- Chargeable dwell days equal `max(0, total_days - free_dwell_days)` using the terminal's `free_dwell_days` in `terminal_tariffs.csv`.
- Base plug-in fee equals `chargeable_days * reefer_plug_daily_usd`.
- Total reefer charge equals `base_plug_in_fee * (1 + power_surcharge_pct / 100)`.
- For live weather risks (high ambient temperature or coastal storms), dispatchers should inspect terminal weather conditions before assigning overflow unshaded bays.

## 3. Alarm Escalation Timelines
- Unacknowledged reefer alarms escalate to the Terminal Duty Engineer after **15 minutes** and to the Regional Cold-Chain Manager after **30 minutes**.
