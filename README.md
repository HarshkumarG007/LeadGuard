# 🛡️ LeadGuard

**LeadGuard 1.0 solved evaluation integrity. LeadGuard 2.0 begins solving decision integrity.**

LeadGuard is an end-to-end machine-learning system for helping municipalities decide which properties to inspect first when the material of a drinking-water service line is unknown.

It combines spatial machine learning, leakage-safe feature engineering, probability calibration, conformal uncertainty, active learning, fairness-aware prioritization, SHAP explainability, and a privacy-conscious API into one reproducible pipeline.

---

## 1. What LeadGuard 1.0 Accomplished

LeadGuard 1.0 established a scientifically characterized decision engine for lead-risk information acquisition. It successfully transitioned the project from a standard predictive modeling pipeline into a rigorous, utility-driven decision engine that explicitly separates **Belief ≠ Policy ≠ Action**. 

By rebuilding the architecture around information-flow integrity, LeadGuard 1.0 proved that a high ML metric can often be evidence of a bad evaluation protocol rather than a good model.

## 2. The Leakage Discovery & Forensic Audit

LeadGuard is deliberately not a story about achieving the highest possible ML score. During early evaluation, an apparently excellent model performance (PR-AUC ≈ 0.99+) was found to be contaminated by label leakage through spatial features.

The original pipeline allowed information derived from known lead labels to influence feature construction before the train/test boundary had been respected. Instead of hiding this, LeadGuard was redesigned around a much stricter principle:
**No evaluation example should receive label-derived information that would not actually be available at prediction time.**

The resulting architecture produced substantially more modest—but much more defensible—performance (Geo PR-AUC ≈ 0.41).

## 3. 1.0 Architecture

The LeadGuard 1.0 architecture structurally separates the expensive analytical components from the hot operational path.

```text
                    ┌───────────────────┐
                    │ Feature Pipeline  │
                    └─────────┬─────────┘
                              ↓
                    ┌───────────────────┐
                    │ Predictive Model  │
                    └─────────┬─────────┘
                              ↓
                    ┌───────────────────┐
                    │ Calibrated Belief │
                    └─────────┬─────────┘
                              ↓
                    ┌───────────────────┐
                    │       EVI         │
                    └─────────┬─────────┘
                              ↓
                    ┌───────────────────┐
                    │ Policy Optimizer  │
                    └─────────┬─────────┘
                              ↓
                    ┌───────────────────┐
                    │ Decision Ledger   │
                    └───────────────────┘
```

## 4. 1.0 Model + Evaluation

LeadGuard 1.0 enforces a strict three-way conceptual split (Train 70% / Cal 15% / Test 15%) and utilizes a geographic holdout to test generalization to unseen spatial regions. The model uses XGBoost for core risk prediction and relies heavily on properly isolated spatial features. 

## 5. 1.0 Performance Table

The reported ~0.41 and ~0.34 results are from the bundled synthetic/sample evaluation.

| Metric                      | Post-fix sample result | Interpretation                    |
| --------------------------- | ---------------------- | --------------------------------- |
| Baseline PR-AUC             | ~0.34                  | Reference model                   |
| XGBoost geographic PR-AUC   | ~0.41                  | Leakage-safe sample result        |
| Absolute PR-AUC improvement | ~0.07                  | XGBoost minus baseline            |
| Relative PR-AUC improvement | ~21%                   | Relative to baseline              |
| Random/Geo gap              | ~12.5%                 | Below 15% project audit threshold |

## 6. Uncertainty, Calibration, Fairness, & Active Learning

LeadGuard 1.0 explicitly models:
* **Calibration**: Models are calibrated via Platt scaling on a held-out calibration set to ensure `P(Lead)` reflects true statistical probability.
* **Conformal Uncertainty**: The system outputs mathematically bounded prediction sets (e.g., `{NotLead, Lead}` when ambiguous).
* **Equity-Aware Ranking**: Demographic and census variables are intentionally omitted from the predictive model and are instead used in a separate fairness-accounting layer to prevent systematically under-inspecting specific neighborhoods.
* **Active Learning**: Feature rebuilding occurs explicitly each time new ground truth is acquired to prevent data staleness.

## 7. Engineering Quality Metrics

| Metric | Status |
|---|---|
| Test coverage | **88.45%** |
| Leakage-audit gap | **12.5%** |
| CI linting (Ruff) | ✅ Passing |
| Automated tests | ✅ Passing locally |

## 8. What Changed After the Audit

```text
suspiciously high metric
           │
           ▼
     forensic audit
           │
           ▼
     leakage found
           │
           ▼
  architecture rebuilt
           │
           ▼
 regression tests added
           │
           ▼
   honest evaluation
           │
           ▼
   defensible metric
```

## 9. 1.0 Limitations

LeadGuard 1.0 currently has important limitations:
* **Optimizer**: The 2-for-2 local search heuristic leaves a 2.3% >1% regret tail under extreme budget tightness (0.14 - 0.18).
* **Storage**: Parquet predicate lookup degrades into sequential scanning under concurrency (API cold start ~1.7s, p50 ~600ms).
* **Calibration**: Point-estimate EVI is highly sensitive to severe miscalibration.
* **Causal utility**: Synthetic experiments do **not** establish real-world causal utility.
* **Observational selection**: Historical inspection data may not represent random observation.
* **Real-world validation**: $EVI_{predicted}$ vs $EVI_{realized}$ remains to be established.

---

## 10. Transition to LeadGuard 2.0

With evaluation integrity solved, LeadGuard 2.0 begins solving decision integrity. We are shifting from a static decision model to a longitudinal observational infrastructure that can measure $EVI_{predicted}$ versus $EVI_{realized}$.

| Area | 1.0 | 2.0 S0 |
| --- | --- | --- |
| Leakage-safe spatial ML | ✅ | Preserved |
| XGBoost | ✅ | Preserved |
| Calibration | ✅ | Preserved |
| Conformal uncertainty | ✅ | Preserved |
| Active learning | ✅ | Preserved |
| Equity-aware ranking | ✅ | Preserved |
| SHAP | ✅ | Preserved |
| Geographic evaluation | ✅ | Preserved |
| **Immutable decision records** | — | **✅ S0** |
| **Decision provenance** | — | **✅ S0** |
| **Temporal outcome barrier** | — | **✅ S0** |
| **Delayed ground-truth linking** | — | **✅ S0** |
| **Shadow-mode environment isolation** | — | **✅ S0** |
| **Dispatch safety boundary** | — | **✅ S0** |
| **Reconstructed realized metrics** | — | **✅ S0** |
| **Monitoring/dashboard layer** | Partial/future | **✅ S0** |
| **Cohort evaluation** | — | **🔜 S1** |

## 11. 2.0 Architecture

```text
    LEADGUARD 1.0 
          │
          ▼ 
  Leakage-safe ML system
          │
  ┌───────┴───────┐
  ▼               ▼
Risk / unc.   Equity / prio
  │               │
  └───────┬───────┘
          ▼
   Inspection queue
          │
          ▼
    Ground truth
          │
          ▼
   Active learning
          │
          ▼
┌─────────────────────┐
│    LEADGUARD 2.0    │
│    S0 FOUNDATION    │
└─────────┬───────────┘
          ▼
   Shadow Decision 
          │
          ▼
 Immutable Decision Log
          │
    30-day barrier 
          │
          ▼
     Observation 
          │
          ▼
   Temporal Linker 
          │
          ▼
Reconstructed Outcomes
          │
          ▼
  Monitoring / EVI
          │
          ▼
  Future Cohort S1
```

## 12. S0 Shadow-Mode Foundation

We have successfully completed Phase S0, establishing the observational infrastructure required for LeadGuard 2.0.

### S0.1 Immutable decisions
Every shadow decision is recorded in an append-only JSONL ledger (`src/leadguard/shadow/decision_recorder.py`). We capture complete provenance (model version, calibration cutoff, optimizer state) and secure the payload with a canonical SHA-256 hash.

### S0.2 Temporal/causal observation barrier
Delayed observations are linked to decisions via `outcome_linker.py`. The linker explicitly enforces a causality barrier: it structurally refuses to join an observation whose `outcome_available_at` falls before or at the original `decision_time`.

### S0.3 Data/environment isolation
The shadow ledgers enforce physical separation by writing to environment-specific namespaces (`data/processed/shadow/synthetic/` vs `production/`). 

### S0.4 Production-dispatch safety invariant
The API (`api/main.py`) implements a hard capability boundary. Unless the environment variable `LEADGUARD_PRODUCTION_MODE=true` is explicitly provided, the system defaults to safely denying operational dispatch, throwing a 403 Forbidden.

### S0.5 Metric reconstruction + monitoring
The dashboard (`src/leadguard/monitoring/dashboard.py`) reconstructs calibration (ECE, Brier) and economic metrics (Predicted vs Realized EVI, Risk-Weighted Regret, Value-Weighted Confusion Matrix) dynamically from the immutable ledgers.

### Shadow-mode execution proof
The full chain was proven via `scripts/run_shadow_cycle.py`, successfully simulating a decision at $T_0$, an inspection at $T_0+5$, an outcome at $T_0+30$, and validating that a temporal join at $T_0+15$ correctly hides the future outcome.

## 13. Current 2.0 Status
The S0 phase is complete. The system is structurally prepared to ingest a real-world shadow cohort without risking operational deployment.

## 14. S1 Roadmap
Phase S1 will initiate a Real-World Shadow Cohort:
* Longitudinal tracking of $EVI_{predicted}$ vs $EVI_{realized}$ on real municipal data.
* Continued calibration monitoring in the wild.
* Identification of actual risk-weighted regret.

## 15. Reproducibility / Audit Artifacts

* **Frozen corpus**: `data/test_fixtures/f7_regret_corpus_v1.json`
* **Adversarial matrix**: `data/processed/f7_matrix.json`

## 16. Limitations and Claims

LeadGuard does **not** claim:
* perfect optimization
* true O(1) storage lookup
* perfect probability calibration
* causal identification from observational data
* guaranteed public-health utility
* that synthetic EVI equals real-world EVI

## 17. Final Research/Engineering Takeaway

LeadGuard is a demonstration of a broader ML engineering principle: trustworthy machine learning is not just about building a powerful model. It is about controlling information flow, measuring uncertainty, validating assumptions, protecting users, and being willing to report a lower number when the lower number is the truth.
