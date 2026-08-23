# LeadGuard 2.0: Phase S1 Cohort Contract

This contract defines the immutable rules of engagement for **LeadGuard Phase S1: Real-World Shadow Cohort**. Its purpose is to prevent post-hoc adjustment of evaluation criteria and to ensure that the longitudinal tracking of $EVI_{predicted}$ versus $EVI_{realized}$ on real municipal data is scientifically sound and operationally safe.

> [!IMPORTANT]
> This contract is frozen *before* real-world shadow data is ingested or analyzed. No definitions, metrics, or counterfactuals can be altered after outcomes are observed.

---

## 1. Cohort Definitions & Scope

To causally evaluate LeadGuard's prioritization against alternatives, the experiment formally distinguishes three populations:

1. **Shadow-Selected Cohort**: Properties the LeadGuard policy explicitly recommended for inspection.
2. **Eligible Universe**: All properties that could legally/operationally have been selected by LeadGuard (the denominator).
3. **Exploration / Control Cohort**: A pre-specified, independently sampled subset generated without LeadGuard's priority score, used to establish the baseline counterfactual observation rate.

### 1.1 Inclusion / Exclusion Rules
* **Included**: Properties actively prioritized for inspection or intervention by the LeadGuard shadow policy between $T_{start}$ and $T_{end}$.
* **Excluded**: Properties excluded by baseline regulatory criteria (e.g., already verified non-lead, emergency dig-ups bypassing normal prioritization). 

### 1.2 Cohort Size & S1 Termination Rule
* **Exposure Minimum**: $N = 5,000$ decisions. 
* **Stopping Rule**: S1 formally closes when **either**:
  1. The minimum 5,000 exposure is reached **AND** the required statistical precision (e.g., min. 150 positive Lead outcomes across strata) is achieved.
  2. A maximum predefined exposure of 20,000 decisions or a time limit of 6 months is reached, regardless of statistical power. 
* *Continuation based on observed effect size to artificially reach significance is strictly prohibited.*

---

## 2. Temporal, Causal, and Feature Semantics

### 2.1 The Grand Causal Invariant
To unambiguously guarantee causal ordering, the following strict inequality must hold for all valid records (unless same-instant decision/observation is genuinely operationally recorded by the municipality):

$$T_{feature\_availability} \le T_{decision} < T_{observation} \le T_{availability}$$

* **Feature Availability ($T_{feature\_availability}$)**: Time the latest input feature was computed/available.
* **Decision Timestamp ($T_{decision}$)**: Time the shadow decision was recorded in the immutable Ledger. 
* **Observation Timestamp ($T_{observation}$)**: Time the physical inspection/intervention occurred in the field.
* **Outcome Availability Timestamp ($T_{availability}$)**: Time the ground-truth label was officially entered into the municipal inventory. Null, timezone-naive, duplicate, or impossible timestamps are explicitly rejected.

### 2.2 Minimum Observation Horizon
* **Operational Horizon**: A decision is evaluated for the primary S1 endpoint if $T_{current} - T_{decision} \ge 30\text{ days}$.
* **Extended Observation**: Outcomes arriving after 30 days are continuously linked for secondary longitudinal analysis, but the primary endpoint remains locked.

---

## 3. Label & Censoring Semantics

### 3.1 Eligible Outcome Labels
* `Lead`, `Copper`, `Galvanized`.

### 3.2 Missing / Unknown Handling
* Field observations resulting in `Unknown` or indeterminate materials are **excluded** from *Information Value* (realized EVI), as they yield no ground truth.
* However, they are **included** in *Operational Cost* calculations (Net Utility). A policy that wastes budget generating ambiguous inspections is penalized.

### 3.3 Censoring Rules
* Decisions that reach the end of the 30-day horizon without an available outcome are marked as **Right-Censored**. They remain in the denominator for cost/efficiency metrics but do not contribute to $EVI_{realized}$.

---

## 4. Provenance & Cryptographic Chain

To guarantee integrity, the raw decision ledger operates as a **Cryptographic Hash Chain**, not merely an append-only JSONL file. 

* Each record must store its own hash and the `previous_record_hash`.
* $H_i = SHA256(record_i || H_{i-1})$.

### 4.1 Required Provenance Fields
* `model_version`, `model_hash`
* `feature_schema_version`
* `calibration_version`, `calibration_hash`
* `policy_version`
* `decision_population_snapshot_hash`: Identifies the exact state of the Eligible Universe at the time of decision.

### 4.2 Action Provenance
* The ledger must explicitly distinguish `shadow_selected` (LeadGuard wanted to inspect) from `actually_inspected` (Municipality actually inspected).
* The `selection_source` and `reason_not_inspected` must be explicitly logged to isolate treatment-selection mechanisms.

---

## 5. Estimands & Analytical Dataset

Once the cohort closes, a **Frozen S1 Analytical Dataset** is generated. The hash of this exact dataset must be published alongside all reported metrics.

S1 is strictly limited to four predefined estimands:

### Estimand A — Predictive (Calibration Stability)
* **Question**: Does predicted risk remain calibrated?
* **Metrics**: Brier Score, ECE, Isotonic slope. 
* **Frozen ECE Definition**: 10 equal-width bins, minimum 50 observations per bin to report, empty bins excluded, aggregated via absolute difference weighted by bin count.

### Estimand B — Economic (Information Value)
* **Question**: Does predicted EVI correlate with realized information value?
* **Metrics**: $EVI_{predicted}$ vs $EVI_{realized}$, excluding `Unknown` from utility but keeping it in the cost.

### Estimand C — Decision (Policy Regret)
* **Question**: Does prioritization outperform predefined baselines?
* **Counterfactual Baselines**: 
  1. **Risk-Only Policy**
  2. **Baseline Municipal Policy**
  3. **Random Policy**: Sampling frame = Eligible Universe; random selection without replacement; seeded deterministically; receives exactly the same budget constraints as LeadGuard.
  4. **Oracle Policy**: *Simulation-only.* Represents the full-information upper-bound. Regret against the Oracle ($U_{oracle} - U_{LeadGuard}$) is strictly an analytical simulation metric, never treated as an observed real-world policy.
* **Metrics**: Observed policy comparison ($U_{LeadGuard} - U_{baseline}$) and Cumulative Risk-Weighted Regret.

### Estimand D — Equity (Distributional Impact)
* **Question**: Does decision quality remain acceptable across predefined cohorts?
* **Metrics**: Equity-adjusted regret, risk-weighted false-negative regret separated by geography and socioeconomic quartile.

---

## 6. Safety & Failure Boundaries

> [!WARNING]
> S1 is a Shadow Mode execution. It does **not** dispatch real interventions.

### 6.1 Calibration Safety Response
* **Monitoring Alert (Early deterioration)**: ECE drifts > 0.08. Triggers alert for data drift review.
* **Soft Stop (Persistent degradation)**: ECE drifts > 0.12 for 7 consecutive days. Triggers a pause on generating new shadow decisions until recalibration.
* **Hard Stop (Catastrophic integrity failure)**: ECE > 0.15 under the frozen estimator triggers an S1 safety halt/review. It does not by itself establish causal or operational failure, but halts the pipeline.

### 6.2 Data Integrity Failures (Hard Stops)
* **Production Mode Leak**: A shadow decision generates a real-world dispatch ticket.
* **Temporal Leakage Detection**: Any outcome violates $T_{decision} < T_{availability}$.
* **Feature Leakage Detection**: Any feature violates $T_{feature\_availability} \le T_{decision}$.
* **Cryptographic Ledger Failure**: A hash-chain mismatch, duplicate/missing ID, or non-monotonic sequence is detected.
* **Identity Mutation**: A decision recorded under model hash X is later evaluated as though it came from model hash Y.
