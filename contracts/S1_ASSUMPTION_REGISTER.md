# S1 Assumption Register

This ledger tracks the fundamental operational assumptions that underlie the evaluation criteria. 

> **Important**: This register explicitly separates the *frozen estimand* (the mathematical definitions) from the *empirical assumptions* (the real-world parameters that could be wrong).

| Assumption | Source | Frozen? | Sensitivity Tested? | Failure Consequence | Owner | Review Date |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| **Inspection Cost ($c$)** | Field Ops Budget | Primary value: $100 | $\pm$ 400% | Negative EVI | LeadGuard Ops | 2026-08-30 |
| **Information Value ($v$)**| Health Economics | Primary value: $5,000 | $\pm$ N/A | Reduced advantage | Health Analytics | 2026-08-30 |
| **Prevalence** | Historical Data | 10% base rate | N/A | Unfavorable denominator | Data Science | 2026-08-30 |
| **False-Negative Cost** | Liability Matrix | High penalty | N/A | Dangerous under-prioritization | Legal/Ops | 2026-08-30 |

## Sensitivity Context
*See `reports/s0/EVI_SENSITIVITY_REPORT.md` for full breakdown.*
The current deployment policy guidance is **VERY ROBUST** against inspection cost variation up to $500 per inspection.
