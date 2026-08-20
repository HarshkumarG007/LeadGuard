# 🛡️ LeadGuard
Leakage-Safe ML for Lead Service-Line Discovery, Uncertainty-Aware Inspection Prioritization, and Equitable Field Deployment

LeadGuard is an end-to-end machine-learning system for helping municipalities decide which properties to inspect first when the material of a drinking-water service line is unknown.

It combines spatial machine learning, leakage-safe feature engineering, probability calibration, conformal uncertainty, active learning, fairness-aware prioritization, SHAP explainability, and a privacy-conscious API into one reproducible pipeline.

## ⚠️ The Most Important Thing About This Project

LeadGuard is deliberately not a story about achieving the highest possible ML score.

During evaluation, an apparently excellent model performance was found to be contaminated by label leakage through spatial features.
The original pipeline could allow information derived from known lead labels to influence feature construction before the train/test boundary had been respected.

That created a deceptively strong result:

```text
BEFORE AUDIT
┌──────────────────────┐
│   Spatial Features   │
│                      │
│   Labels available   │
│       too early      │
└──────────┬───────────┘
           │
           ▼
   Model Evaluation
           │
           ▼
    PR-AUC ≈ 0.99+
           │
           ▼
  🚨 Suspiciously high
```

Instead of hiding this, LeadGuard was redesigned around a much stricter principle:
**No evaluation example should receive label-derived information that would not actually be available at prediction time.**

The resulting architecture produces substantially more modest—but much more defensible—performance:

```text
AFTER AUDIT
┌──────────────────────┐
│     Raw Features     │
└──────────┬───────────┘
           │
           ▼
      Split FIRST
           │
 ┌─────────┼─────────┐
 ▼         ▼         ▼
TRAIN     CAL      TEST
 │         │         │
 │         │         │
 └────┐    │    ┌────┘
      ▼    ▼    ▼
 Leakage-safe evaluation
           │
           ▼
   Geo PR-AUC ≈ 0.41
```

This is one of the central engineering lessons of LeadGuard:
**A high metric can be evidence of a good model—or evidence of a bad evaluation protocol.**

---

## 📌 Current Project Status

| Area | Current status |
| --- | --- |
| End-to-end pipeline | ✅ Implemented |
| Spatial feature engineering | ✅ Leakage-aware |
| Label-leakage regression tests | ✅ Implemented |
| Train/calibration/test separation | ✅ Implemented |
| XGBoost | ✅ Implemented |
| Probability calibration | ✅ Implemented |
| Binary conformal uncertainty | ✅ Implemented |
| Pseudo-ensemble | ✅ Removed |
| Active learning | ✅ Iterative retraining |
| Spatial features during active learning | ✅ Rebuilt each round |
| Equity-aware prioritization | ✅ Implemented |
| SHAP explainability | ✅ Implemented |
| API | ✅ FastAPI |
| Dashboard | ✅ Streamlit |
| API address exposure | ✅ Removed from public priority queue |
| CI linting | ✅ Ruff |
| Automated tests | ✅ Passing locally |
| Test coverage | **88.45%** |
| Leakage-audit gap | **12.5%** |
| Sample geographic PR-AUC | **~0.41** |
| Sample baseline PR-AUC | **~0.34** |

### Important interpretation
The reported ~0.41 and ~0.34 results are from the bundled synthetic/sample evaluation.
They are useful for:
- reproducibility,
- regression testing,
- demonstrating the pipeline,
- comparing architecture changes.

They are **not** evidence that LeadGuard achieves the same performance on real municipal data.
The repository's data card explicitly identifies the sample dataset as synthetic and cautions that sample metrics do not represent real Chicago performance. ([GitHub](https://github.com/HarshkumarG007/LeadGuard/blob/main/docs/data_card.md))

---

## 🎯 1. The Problem

Municipal water systems may contain hundreds of thousands of service lines whose material is unknown.
For each property, the important question is deceptively simple:
*Is the service line made of lead?*

But the operational question is much harder:
*Given a limited inspection budget, which properties should inspectors visit first?*

Suppose a city has:
- 500,000 properties
- 100,000 unknown service lines
- 5,000 inspections available

Inspecting randomly wastes a large fraction of the budget.
Inspecting only the highest-risk predictions introduces another problem:
- uncertainty may be ignored,
- under-inspected neighborhoods may remain under-inspected,
- the model may repeatedly sample the same spatial regions,
- newly discovered labels may not be incorporated efficiently.

LeadGuard therefore treats the task as **decision optimization**, not merely binary classification.

## 🧠 2. What LeadGuard Actually Predicts

At the ML layer, the problem is binary classification:
- `y = 1 → Lead`
- `y = 0 → NotLead`

The negative class includes non-lead materials such as:
- Copper
- Galvanized

Unknown materials are not treated as known negative examples.

The model estimates:
$$P(\text{Lead} \mid X)$$
where $X$ contains information that would be legitimately available for the property being scored.

The resulting probability is useful—but insufficient.
A municipality does not need a probability table.
It needs an inspection queue.

Therefore LeadGuard turns predictions into an operational priority score incorporating:
- Risk
- Uncertainty
- Equity

## 🏗️ 3. System Architecture

```text
    ┌──────────────────────┐
    │     Data Sources     │
    │                      │
    │   Water inventory    │
    │  Property assessor   │
    │    OpenStreetMap     │
    │      Census ACS      │
    └──────────┬───────────┘
               │
               ▼
    ┌──────────────────────┐
    │    Data Cleaning     │
    │       clean.py       │
    └──────────┬───────────┘
               │
               ▼
    ┌──────────────────────┐
    │ Feature Engineering  │
    │     features.py      │
    └──────────┬───────────┘
               │
      ┌────────┴────────┐
      │                 │
      ▼                 ▼
 Raw/non-label    Label-dependent
   features      spatial features
      │                 │
      │       ONLY from permitted
      │        reference labels
      │                 │
      ▼                 ▼
    ┌──────────────────────┐
    │    Model Training    │
    │       XGBoost        │
    └──────────┬───────────┘
               │
               ▼
    ┌──────────────────────┐
    │     Calibration      │
    │  Held-out CAL split  │
    └──────────┬───────────┘
               │
      ┌────────┼────────┐
      ▼        ▼        ▼
 Probability Conformal   SHAP
 estimates  uncertainty explanations
      │        │        │
      └────────┼────────┘
               ▼
    ┌──────────────────────┐
    │ Priority Generation  │
    │                      │
    │  Risk + Uncertainty  │
    │       + Equity       │
    └──────────┬───────────┘
               │
        ┌──────┴──────┐
        ▼             ▼
     FastAPI      Streamlit
       API           UI
        │             │
        └──────┬──────┘
               ▼
        Human inspection
               │
               ▼
        New ground truth
               │
               ▼
        Active learning
               │
               └──────► Model
```

The architecture document formalizes the system as separate data, modeling, prioritization, API, and evaluation components. ([GitHub](https://github.com/HarshkumarG007/LeadGuard/blob/main/LeadGuard_Architecture.md))

## 🔥 4. The Leakage Problem

This deserves its own section because it is the most important technical lesson in the project.

### What is label leakage?
Suppose we want to predict whether property A has lead.
We know that several nearby properties have already been inspected.

Imagine:
- Property B (Lead) ●
- Property A (unknown) ●
- Property C (Lead) ●

A feature such as:
`neighbor_lead_rate = nearby_known_lead / nearby_known_properties`
can be extremely predictive.

That is perfectly legitimate if the nearby labels were already known at prediction time.
But it becomes leakage if the feature-generation process uses labels from the evaluation set—or labels that would not have been available when the model is supposed to make its prediction.

### ❌ The Dangerous Pipeline

The original conceptual failure looked like:
```text
      FULL DATASET
           │
           ▼
Build label-dependent spatial features
           │
           │ ← labels from everywhere can influence features
           ▼
    Split train/test
           │
           ▼
      Train model
           │
           ▼
       Evaluate
```

The model can indirectly see information about the evaluation distribution.
That makes the test set no longer truly independent.

### ✅ The Correct Pipeline

LeadGuard now follows:
```text
        RAW DATA
           │
           ▼
      DEFINE SPLIT
           │
      ┌────┴─────┐
      ▼          ▼
    TRAIN       TEST
      │          │
      │          │
      │          └──── Test labels remain hidden
      ▼
Build label-dependent features
using only permitted TRAIN labels
      │
      ▼
  Train model
      │
      ▼
Generate TEST features
using TRAIN labels only
      │
      ▼
   Evaluate
```

Conceptually:
$$X_i^{spatial} = f(i, \mathcal{L}_{train})$$
where:
- $i$ is the property being scored,
- $f$ is the spatial feature function,
- $\mathcal{L}_{train}$ is the set of labels legitimately available to the model.

Critically:
$$y_{test} \notin X_{test}$$
and:
$$y_{test} \not\rightarrow X_{train}$$

## 🧪 5. Leakage Regression Tests

A model can pass ordinary unit tests while still leaking information.
Therefore LeadGuard includes tests specifically designed to attack the evaluation pipeline.

**Test A — Test-label invariance**
Change test labels.
Then regenerate features.
Expected:
- TRAIN FEATURES: unchanged
- TEST FEATURES: unchanged

If test labels change the features, the pipeline is leaking.

**Test B — Training-label perturbation**
Randomize the training labels.
Expected:
- model performance ↓ toward prevalence baseline

If performance remains spectacular after destroying the training signal, something is suspicious.

**Test C — Geographic isolation**
Verify that geographic holdout features do not accidentally consume geographic test labels.
This is particularly important because spatial ML can hide leakage in seemingly innocent aggregate features.

## 📊 6. Why Random Splits Are Not Enough

Imagine this:
- Train: ● ● ● ● ● ● ● ● ● ● ● ●
- Test:  ●          ●

A random split can place geographically adjacent properties into both sets.
The model can therefore learn highly local patterns.

A geographic holdout is harder:
```text
TRAIN REGION
████████████████
████████████████
████████████████

TEST REGION
░░░░░░░░░░░░░░░░
░░░░░░░░░░░░░░░░
```

The question becomes:
*Can the model generalize to a different geographic area?*
That is much closer to the operational problem of deploying a prioritization model into neighborhoods where the model has fewer direct labels.

## 🔬 7. Evaluation Design

LeadGuard uses a strict three-way conceptual split:

```text
        DATA
         │
   ┌─────┼─────┐
   ▼     ▼     ▼
 TRAIN  CAL   TEST
  70%   15%   15%
```

**TRAIN**
Used for:
- model fitting,
- learning parameters,
- hyperparameter optimization,
- training-time spatial reference labels.

**CAL**
Used exclusively for:
- probability calibration,
- conformal calibration where applicable.

**TEST**
Used only for:
- final evaluation,
- headline performance reporting.

The test set should not influence model selection.

## 🎯 8. Why Calibration Matters

A classifier can rank properties correctly without producing trustworthy probabilities.

For example:
- Property A → 0.91
- Property B → 0.72
- Property C → 0.51

The ordering might be useful.
But what does `0.72` actually mean?

If the model is calibrated, then among properties receiving approximately 0.72 predictions, roughly 72% should be positive over a sufficiently large comparable population.
This matters because inspection planning is a resource-allocation problem.

## 📐 9. Calibration Architecture

LeadGuard separates:
```text
      TRAIN
        │
        ▼
     XGBoost
        │
        ▼
 raw probability
        │
        ▼
 CALIBRATION SET
        │
        ▼
Platt / sigmoid calibration
        │
        ▼
calibrated P(Lead)
```

The calibrated model is what the serving layer should use for probability-based decisions.
The modern scikit-learn API requires a frozen/pre-fitted estimator when calibrating an already-trained model in the newer versions supported by the project.

## 📏 10. Calibration Metrics

A serious evaluation should include:

**Brier Score**
$$BS = \frac{1}{N}\sum_{i=1}^{N}(p_i-y_i)^2$$
Lower is better.
It measures probabilistic accuracy.

**Log Loss**
$$-\frac{1}{N} \sum_i \left[ y_i\log(p_i) + (1-y_i)\log(1-p_i) \right]$$
Lower is better.
Log loss strongly penalizes confident incorrect predictions.

**Expected Calibration Error**
ECE divides predictions into confidence bins and compares:
*average confidence vs observed frequency*
Conceptually:
$$ECE = \sum_b \frac{|B_b|}{N} |\operatorname{acc}(B_b)-\operatorname{conf}(B_b)|$$
Lower is better.

### ⚠️ Calibration Claim Discipline
LeadGuard distinguishes:
*"The model is calibrated using a held-out calibration set."*
from:
*"The model is perfectly calibrated."*

The first is an architectural fact.
The second requires empirical evidence.
Therefore the README should report Brier score, log loss, ECE, and reliability curves whenever those final metrics have been generated.

## 🧮 11. Uncertainty Quantification

Risk alone is not enough.

Consider:
- Property A: P(Lead) = 0.51
- Property B: P(Lead) = 0.99

Both are technically candidates.
But they represent very different states of knowledge.

LeadGuard therefore separates:
- **RISK**: How likely is lead?
- **UNCERTAINTY**: How ambiguous is the prediction?
- **EQUITY**: Who is being systematically missed?

## 🔐 12. Binary Conformal Semantics

LeadGuard's prediction universe is explicitly:
`MATERIALS = ["NotLead", "Lead"]`
rather than pretending the model is predicting three independent classes.

The mapping is:
- `0 → NotLead`
- `1 → Lead`

This is important because the actual model objective is binary.

## 📦 13. What Conformal Prediction Gives You

Instead of always returning `Lead`, the model can return a prediction set such as:
`{Lead}`
or:
`{NotLead}`
or, when uncertain:
`{NotLead, Lead}`

The third result communicates:
*The system does not have enough statistical evidence to confidently resolve the class at the selected coverage level.*

That is operationally useful.
An uncertain property can become a valuable inspection candidate.

## 📈 14. Predictive Ambiguity

LeadGuard also exposes a continuous ambiguity score derived from the calibrated lead probability.

A simple formulation is:
$$U(p) = 1-|2p-1|$$

Therefore:
- p = 0.00 → U = 0.00
- p = 0.10 → U = 0.20
- p = 0.50 → U = 1.00
- p = 0.90 → U = 0.20
- p = 1.00 → U = 0.00

So `p ≈ 0.5` means maximum ambiguity.
This is intentionally different from pretending that noisy model replicas constitute an ensemble.

## 🚫 15. The Fake Ensemble Was Removed

An earlier uncertainty implementation perturbed predictions using Gaussian noise and interpreted disagreement as ensemble uncertainty.

That was removed.
Why?
Because adding random noise to a single model does not magically create a statistically meaningful ensemble.

LeadGuard now distinguishes:
- **CALIBRATED PROBABILITY** → predictive ambiguity
- **CONFORMAL PREDICTION** → prediction set / coverage
- **MODEL ENSEMBLES** → not simulated through arbitrary noise

This is a deliberate methodological cleanup.

## 🤖 16. Why XGBoost?

LeadGuard uses XGBoost for the main predictive model.

The dataset is fundamentally tabular:
- property characteristics,
- categorical fields,
- spatial variables,
- engineered numerical features,
- missing values,
- nonlinear relationships.

Tree ensembles are a natural fit.
The methodology also emphasizes CPU-friendly training and inference. ([GitHub](https://github.com/HarshkumarG007/LeadGuard/blob/main/docs/methodology.md))

## 🌲 17. Model Features

Examples of legitimate predictive information include:

**Property features**
- year_built
- lot_size_sqft
- property characteristics
- building characteristics

**Spatial infrastructure features**
- distance to nearest hydrant
- spatial neighborhood structure
- geographic context

**Label-derived spatial features**
These are allowed only when generated from an explicitly permitted reference label set.
Examples include:
- neighbor_lead_rate
- knn_lead_rate
- dist_to_nearest_known_lead_m

The important part is not the feature name.
The important part is: **Where did its labels come from?**

## 🗺️ 18. Spatial Features: The Subtle Part

Spatial features are powerful because infrastructure is geographically correlated.
If one neighborhood has older service lines, nearby properties may have similar characteristics.

But spatial correlation creates an evaluation hazard.

Consider:
```text
Known Lead        Unknown
    ●               ○
     \             /
      \           /
       \         /
        ●───────●
            Known Lead
```
A nearest-known-lead feature can be extremely predictive.

That is useful operationally after inspections have actually happened.
But during historical evaluation, it must represent only information that would have been known at that point.

LeadGuard therefore treats label-dependent spatial features as stateful information, not static columns.

## 🔄 19. Active Learning

Normal supervised learning assumes:
`dataset → train once → deploy`

But LeadGuard operates in a world where inspections create new labels.
Therefore:

```text
     Initial labels
           │
           ▼
      Train model
           │
           ▼
Score unknown properties
           │
           ▼
   Choose inspections
           │
           ▼
 Receive ground truth
           │
           ▼
 Expand labeled set
           │
           ▼
Rebuild label-dependent
   spatial features
           │
           ▼
       Retrain
           │
           ▼
        Repeat
```

This is active learning.

## 🔁 20. The Critical Active-Learning Fix

The active-learning loop must not do this:
`Train once ↓ Predict repeatedly`
That would merely simulate changing rankings.

Instead:
```text
      Round 0
         ↓
        Fit
         ↓
   Acquire labels
         ↓
      Round 1
         ↓
  Rebuild features
         ↓
     Fit again
         ↓
   Acquire labels
         ↓
      Round 2
         ↓
  Rebuild features
         ↓
     Fit again
```

The newly acquired labels can change spatial features.
Therefore the feature matrix itself is part of the evolving state.

## 🧪 21. Active-Learning Strategies

LeadGuard compares multiple acquisition policies.

**Random**
Select properties randomly.
Purpose: Establish the baseline rate of learning.

**Highest Risk**
Select: $\operatorname{arg\,top}_k P(\text{Lead})$
Purpose: Maximize immediate expected lead discoveries.

**Highest Uncertainty**
Select properties where predictions are most ambiguous.
Purpose: Learn where the model knows least.

**Risk + Uncertainty**
Combine high probability of lead + high uncertainty.
Purpose: Balance exploitation and information gain.

**Risk + Uncertainty + Equity**
Add an equity component.
Purpose: Avoid repeatedly concentrating inspections in already well-covered areas.

**Oracle**
Use unavailable ground truth to establish an upper-bound benchmark.
The oracle is not deployable.
It answers: *"How well could this acquisition process do if we knew the answers in advance?"*

## ⚖️ 22. Equity-Aware Prioritization

A purely predictive model can unintentionally reproduce historical inspection patterns.

Suppose:
- Neighborhood A: many previous inspections
- Neighborhood B: very few inspections

Even if both have similar predicted risk, a purely risk-driven queue may repeatedly prioritize A.
LeadGuard therefore maintains an equity accounting layer.

The core idea is:
$$\text{Priority} = \alpha \text{Risk} + \beta \text{Uncertainty} + \gamma \text{Equity}$$
where the equity term reflects inspection coverage relative to model-estimated risk share.

The methodology explicitly keeps demographic/income fields out of the predictive feature matrix and uses them in a separate fairness/reference pathway. ([GitHub](https://github.com/HarshkumarG007/LeadGuard/blob/main/docs/methodology.md))

## 🚫 23. Sensitive Variables Are Not Model Features

LeadGuard deliberately avoids directly feeding fields such as:
- race
- income
- demographic indicators

into the predictive model.
This prevents the model from simply learning a shortcut:
`demographic variable ↓ prediction`

Instead:
```text
┌──────────────┐
│ Model inputs │
└──────┬───────┘
       │
       ▼
    XGBoost
       │
       ▼
    P(Lead)
```

while fairness accounting remains separate:
```text
Census / demographic context
       │
       ▼
Fairness reference
       │
       ▼
Equity accounting
       │
       ▼
Priority adjustment
```
This separation is an important architectural boundary.

## 📊 24. Fairness Evaluation

Fairness should not mean: *"The model must predict every group identically."*
That can be mathematically inappropriate.

Instead LeadGuard evaluates whether prioritization produces unacceptable disparities.
Relevant measurements can include:
- false-negative-rate disparity,
- coverage by income quartile,
- inspection allocation,
- risk capture,
- uncertainty coverage,
- geographic inspection distribution.

The sample data card identifies the automated demographic-leakage test as a CI constraint. ([GitHub](https://github.com/HarshkumarG007/LeadGuard/blob/main/docs/data_card.md))

## 🧭 25. Geographic Holdout

LeadGuard evaluates both conventional and geographic generalization.

The geographic question is:
*Does the model still work when evaluated on a region whose labels were not used to build the model's label-dependent spatial features?*

This is harder than random splitting.
It is also more relevant to deployment because a municipality may need to prioritize areas where direct inspection evidence is sparse.

## 📉 26. Post-Fix Performance Audit

The remediation produced a dramatically different result from the original headline.

**Before**
Synthetic evaluation PR-AUC ≈ 0.99+
This result was treated as suspicious after the leakage audit.

**After**
Leakage-safe sample evaluation:
- Baseline PR-AUC ≈ 0.34
- XGBoost Geo PR-AUC ≈ 0.41

Absolute difference:
$$0.41 - 0.34 = 0.07$$

Relative difference:
$$\frac{0.41 - 0.34}{0.34} \approx 20.6\%$$

So the appropriate description is:
**Approximately +0.07 absolute PR-AUC, or approximately +21% relative to the baseline PR-AUC, on the bundled sample evaluation.**

This is not a claim of production performance.

## 🔍 27. Random-vs-Geographic Gap

The post-fix evaluation reports approximately:
`Random/Geo PR-AUC gap ≈ 12.5%`

LeadGuard uses `15%` as its project-specific leakage-audit threshold.
Therefore:
**The observed 12.5% gap is below LeadGuard's predefined 15% audit threshold.**

It should not be interpreted as:
*"12.5% proves that leakage is impossible."*

The stronger evidence is the combination of:
- split-first feature construction,
- label invariance tests,
- geographic holdout,
- train/calibration/test separation,
- active-learning feature rebuilding.

## 📈 28. Current Metrics Snapshot

| Metric | Post-fix sample result | Interpretation |
| --- | --- | --- |
| Baseline PR-AUC | ~0.34 | Reference model |
| XGBoost geographic PR-AUC | ~0.41 | Leakage-safe sample result |
| Absolute PR-AUC improvement | ~0.07 | XGBoost minus baseline |
| Relative PR-AUC improvement | ~21% | Relative to baseline |
| Random/Geo gap | ~12.5% | Below 15% project audit threshold |
| Test coverage | 88.45% | Above 80% CI requirement |

**Important**
These numbers are sample/synthetic evaluation results.
They should not be interpreted as:
- real-world Chicago accuracy,
- production generalization,
- guaranteed discovery rate,
- causal evidence,
- regulatory certification.

*(Placeholders for final Brier score, log loss, ECE, conformal coverage, and active-learning benchmark curves will be populated from generated artifacts in future runs).*

## 🧠 29. Why PR-AUC?

Lead detection is naturally an imbalanced classification problem.
Accuracy can be misleading.

For example:
- 95% NotLead
- 5% Lead

A model predicting `NotLead` for everyone achieves 95% accuracy while finding no lead lines.

PR-AUC focuses more directly on the positive class and the precision/recall tradeoff.
That makes it a better primary metric for the inspection-discovery problem.

## 🧮 30. Baseline Models

LeadGuard does not evaluate XGBoost in isolation.
Baselines provide context.

A useful baseline answers:
*"Is the complex model actually doing something useful?"*

The project includes baseline evaluation alongside the advanced model.
The post-fix sample comparison is approximately:
- Baseline: 0.34 PR-AUC
- XGBoost: 0.41 PR-AUC

The correct conclusion is therefore not: *"XGBoost is amazing."*
It is: **The leakage-safe XGBoost pipeline captures additional ranking signal over the baseline on the bundled sample evaluation.**

## 🔬 31. Explainability

A municipality should not receive:
`Property 12345 P(Lead) = 0.87`
and be told: *"Trust the model."*

LeadGuard integrates SHAP-based explanations.

Conceptually:
$$f(x) = \phi_0 + \sum_i\phi_i$$
where each $\phi_i$ represents a feature's contribution to the prediction.

A planner can therefore see a simplified explanation such as:
```text
P(Lead) = 0.87
Top contributors:
+ Older construction year
+ Spatial lead prevalence
+ Property characteristics
+ Geographic context
- Recent construction indicator
```

The methodology specifies `TreeExplainer` for efficient feature attribution. ([GitHub](https://github.com/HarshkumarG007/LeadGuard/blob/main/docs/methodology.md))

## 🛡️ 32. API Privacy

The public inspection-priority endpoint should not expose unnecessary personally identifying/location-sensitive information.

The public queue therefore focuses on identifiers and decision information rather than returning raw street addresses.

Conceptually:
```json
{
  "property_id": "P12345",
  "p_lead": 0.87,
  "uncertainty": 0.31,
  "priority_score": 0.82
}
```
rather than:
```json
{
  "property_id": "P12345",
  "address": "123 Example Street ...",
  ...
}
```

This is an important principle:
**A model can be technically correct while its interface still exposes too much information.**
Privacy therefore belongs in the architecture, not only in the database.

## 🖥️ 33. Application Layer

LeadGuard has two primary interfaces.

**FastAPI**
Used for:
- prediction,
- priority queue generation,
- model serving,
- structured programmatic access.
The repository exposes the FastAPI application under `api/`.

**Streamlit**
Used for:
- interactive exploration,
- prioritization,
- model explanations,
- demonstration.
The repository exposes the dashboard under `app/`.

## 📂 34. Repository Structure

```text
LeadGuard/
│
├── api/
│   ├── main.py
│   └── schemas.py
│
├── app/
│   └── streamlit_app.py
│
├── configs/
│   └── train.yaml
│
├── data/
│   ├── raw/
│   ├── interim/
│   ├── processed/
│   └── sample/
│
├── docs/
│   ├── methodology.md
│   └── data_card.md
│
├── models/
│   └── xgboost/
│       ├── reports/
│
├── src/
│   └── leadguard/
│       │
│       ├── data/
│       │   ├── clean.py
│       │   ├── download.py
│       │   ├── features.py
│       │   └── validation.py
│       │
│       ├── evaluation/
│       │   ├── explainability.py
│       │   ├── fairness.py
│       │   └── metrics.py
│       │
│       ├── models/
│       │   ├── active_learning.py
│       │   ├── baseline.py
│       │   ├── uncertainty.py
│       │   └── xgboost_model.py
│       │
│       └── utils/
│           ├── geospatial.py
│           └── seed.py
│
├── tests/
│   ├── unit/
│   └── integration/
│
├── .github/
│   └── workflows/
│       └── ci.yml
│
├── Dockerfile
├── Makefile
├── pyproject.toml
├── LeadGuard_Architecture.md
├── PROGRESS.md
└── README.md
```

## 🚀 35. Installation

**Clone**
```bash
git clone https://github.com/HarshkumarG007/LeadGuard.git
cd LeadGuard
```

**Create environment**
Linux/macOS
```bash
python3 -m venv .venv
source .venv/bin/activate
```
Windows
```powershell
python -m venv .venv
.venv\Scripts\Activate.ps1
```

**Install**
```bash
pip install -e ".[dev]"
```

## ⚡ 36. Quick Start

The repository includes a synthetic sample dataset intended for reproducible demonstration.
The current data card describes it as a 7,500-row synthetic dataset. ([GitHub](https://github.com/HarshkumarG007/LeadGuard/blob/main/docs/data_card.md))

Run:
```bash
python -m leadguard.data.clean \
  --input data/sample \
  --output data/interim/sample_interim.parquet
```

Then:
```bash
python -m leadguard.data.features \
  --input data/interim/sample_interim.parquet \
  --output data/processed/features_sample.parquet
```

Train the baseline:
```bash
python -m leadguard.models.baseline \
  --config configs/train.yaml \
  --features data/processed/features_sample.parquet
```

Train XGBoost:
```bash
python -m leadguard.models.xgboost_model \
  --config configs/train.yaml \
  --features data/processed/features_sample.parquet
```

Run uncertainty:
```bash
python -m leadguard.models.uncertainty \
  --features data/processed/features_sample.parquet \
  --sample
```

Run fairness:
```bash
python -m leadguard.evaluation.fairness \
  --sample
```

Run active learning:
```bash
python -m leadguard.models.active_learning \
  --sample
```

Run explainability:
```bash
python -m leadguard.evaluation.explainability \
  --sample
```

## 🧰 37. Makefile Workflow

For the sample pipeline:
```bash
make train-sample
```

For serving:
```bash
make serve
```

For the dashboard:
```bash
make dashboard
```

The repository's `Makefile` provides the convenience wrappers around the underlying pipeline stages. ([GitHub](https://github.com/HarshkumarG007/LeadGuard/blob/main/Makefile))

## 🧪 38. Testing

Run the entire suite:
```bash
python -m pytest tests/ -v
```

Run with coverage:
```bash
python -m pytest \
  tests/ \
  --cov=src \
  --cov-report=term-missing \
  --cov-fail-under=80
```

Current validated repository coverage: **88.45%**
This exceeds the project's CI requirement: **80%**

## 🔬 39. Targeted Tests

Leakage
```bash
python -m pytest tests/unit/test_leakage.py -v
```

Active learning
```bash
python -m pytest tests/unit/test_active_learning.py -v
```

Uncertainty
```bash
python -m pytest tests/unit/test_uncertainty.py -v
```

Integration
```bash
python -m pytest tests/integration/test_pipeline.py -v
```

## 🧹 40. Code Quality

LeadGuard uses Ruff for linting and formatting.

Run:
```bash
python -m ruff check .
```

Format:
```bash
python -m ruff format .
```

Verify formatting:
```bash
python -m ruff format --check .
```

The desired CI state is:
```text
Ruff         PASS
Formatting   PASS
Tests        PASS
Coverage     >= 80%
```

## 🔄 41. Reproducibility

LeadGuard is designed so that the major evaluation stages can be regenerated.
The basic reproducibility chain is:
`raw/sample data ↓ clean ↓ feature generation ↓ baseline ↓ XGBoost ↓ calibration ↓ uncertainty ↓ fairness ↓ active learning ↓ explainability`

This is preferable to storing only a final score because every important result can be traced back to a pipeline stage.

## 🧬 42. Data Sources

The production-oriented architecture is designed around several data sources.

**Water service-line inventory**
Provides:
- property/service-line identifiers,
- known material labels,
- geographic information.

**Property assessor**
Provides:
- property characteristics,
- construction information,
- parcel attributes.

**OpenStreetMap**
Used for infrastructure-derived spatial information such as nearest hydrant distance.

**Census ACS**
Used for fairness/equity reference information rather than predictive model inputs.

The current data card documents these sources and their known quality issues. ([GitHub](https://github.com/HarshkumarG007/LeadGuard/blob/main/docs/data_card.md))

## ⚠️ 43. Data Quality

Real infrastructure data is messy.
Known issues include:
- unknown service-line materials,
- self-reported material labels,
- geocoding errors,
- missing construction years,
- imperfect property classifications.

The system therefore treats data cleaning as a first-class ML stage.

## 🧹 44. Cleaning Pipeline

Conceptually:
```text
          Raw records
               │
               ▼
     Normalize identifiers
               │
               ▼
      Normalize addresses
               │
               ▼
      Resolve duplicates
               │
               ▼
Validate geographic coordinates
               │
               ▼
    Normalize material labels
               │
               ▼
      Handle missing values
               │
               ▼
  Validated intermediate dataset
```

## 🧮 45. Missing Data

Missingness is not automatically equivalent to zero.
For example:
`year_built = missing`
does not mean:
`year_built = 0`

Feature engineering therefore applies appropriate imputation/handling rules rather than blindly converting missing values into meaningful measurements.

## 🔐 46. Information-Flow Security Model

A useful way to understand LeadGuard is as an information-flow problem.
There are three categories of information:

**Category A — Always permissible**
- property attributes
- geographic coordinates
- infrastructure features
- historical metadata

**Category B — Conditionally permissible**
- known nearby labels
- neighbor lead rate
- distance to known lead
- KNN label statistics

These are permissible **only** if the labels belong to the legitimate reference set available at prediction time.

**Category C — Forbidden during test evaluation**
- test labels
- future inspection results
- labels from the held-out region

The core rule is:
`TEST LABEL │ X │ FEATURE MATRIX`

## 🧠 47. The Deep Design Principle

The most important architectural idea in LeadGuard is:
**Features are not inherently safe or unsafe. Their provenance determines whether they are safe.**

For example:
`neighbor_lead_rate`
is not automatically leakage.

It depends on:
- Which neighbors?
- Which labels?
- When were those labels known?
- Which split do they belong to?

This is why LeadGuard's feature-generation architecture is split-aware.

## 🧪 48. What We Learned From the Premortem

Before implementation, several failure modes were identified.

**Failure mode 1: Spatial label leakage**
Mitigation: split-first architecture, reference-only labels, invariance tests.

**Failure mode 2: Calibration on the test set**
Mitigation: dedicated calibration partition.

**Failure mode 3: Fake uncertainty**
Mitigation: remove Gaussian pseudo-ensemble, use calibrated probability ambiguity, use conformal sets.

**Failure mode 4: Fake active learning**
Mitigation: retrain every round, rebuild label-dependent features every round.

**Failure mode 5: Privacy leakage through API**
Mitigation: remove raw address from public priority responses.

**Failure mode 6: Metric inflation**
Mitigation: geographic holdout, independent test evaluation, explicit pre/post audit.

## 🩺 49. Autopsy of the Original Evaluation

The project intentionally treats the first model evaluation like a forensic investigation.

The symptom: `PR-AUC ≈ 1.0`
The question: *"Why is this model so good?"*
Not: *"How do we put 1.0 in the README?"*

Investigation revealed:
`spatial correlation + label-dependent features + incorrect feature-generation timing ↓ information crossing evaluation boundaries`

The remedy was architectural rather than cosmetic.
That distinction matters.

## 💀 50. What the Critic Should Attack

A serious reviewer should ask:

**"Are the spatial features leakage-safe?"**
Answer: They are generated after split definition and label-dependent features consume only permitted reference labels.

**"Can test labels change test features?"**
Answer: Regression tests explicitly test this invariant.

**"Are probabilities calibrated?"**
Answer: Calibration is trained on a dedicated held-out calibration partition. Empirical Brier/log-loss/ECE results should be reported alongside the implementation claim.

**"Does active learning actually learn?"**
Answer: The simulation retrains the model and rebuilds label-dependent spatial features after every acquisition round.

**"Is uncertainty a fake ensemble?"**
Answer: No. The Gaussian perturbation pseudo-ensemble was removed.

**"Is the fairness variable in the model?"**
Answer: Sensitive/equity information is kept outside the predictive feature matrix and used in separate fairness/equity accounting.

**"Does the API expose addresses?"**
Answer: The public priority queue schema no longer includes the address field.

**"Are the 0.41 results production results?"**
Answer: No. They are results from the bundled synthetic/sample evaluation.

## 🧱 51. What LeadGuard Does Not Claim

LeadGuard does not currently claim:
- production-level accuracy,
- nationwide generalization,
- causal inference,
- guaranteed lead detection,
- perfect calibration,
- perfect fairness,
- elimination of all possible leakage,
- regulatory certification,
- autonomous inspection decisions.

Instead, it claims something more defensible:
**LeadGuard is a reproducible research/portfolio system demonstrating how a leakage-sensitive spatial ML problem can be engineered, evaluated, audited, explained, and converted into an uncertainty- and equity-aware inspection prioritization workflow.**

## 📚 52. Documentation

The repository contains dedicated documentation for:
- System architecture: `LeadGuard_Architecture.md`
- Modeling methodology: `docs/methodology.md`
- Data card: `docs/data_card.md`
- Project progress: `PROGRESS.md`

The methodology document describes the binary formulation, XGBoost model, conformal uncertainty, equity-aware prioritization, active learning, and SHAP explainability. ([GitHub](https://github.com/HarshkumarG007/LeadGuard/blob/main/docs/methodology.md))
The data card documents the sample dataset, production-oriented sources, quality issues, fairness constraints, and limitations. ([GitHub](https://github.com/HarshkumarG007/LeadGuard/blob/main/docs/data_card.md))

## 🗺️ 53. Development Roadmap

**Completed**
- Data cleaning
- Feature engineering
- Spatial modeling
- XGBoost
- Baselines
- Geographic evaluation
- Calibration architecture
- Binary conformal uncertainty
- Active learning
- Equity-aware prioritization
- SHAP explanations
- FastAPI
- Streamlit
- Leakage regression tests
- API privacy regression
- CI quality gates
- Post-fix audit

**Future work**
- Larger real-world labeled dataset
- Field-validation campaign
- Temporal validation
- City-to-city transfer evaluation
- Confidence intervals for headline metrics
- Bootstrap uncertainty around PR-AUC
- More rigorous calibration reporting
- Production model registry
- Monitoring for feature drift
- Monitoring for label drift
- Model versioning
- Human-in-the-loop feedback capture
- Spatial-temporal active learning
- Cost-sensitive acquisition optimization

## 🧭 54. Recommended Production Evolution

A production system should evolve from:
`Static dataset ↓ One-time model ↓ One-time evaluation`

toward:
```text
    ┌──────────────┐
    │ Data sources │
    └──────┬───────┘
           ▼
    Data validation
           │
           ▼
    Feature pipeline
           │
           ▼
     Model registry
           │
           ▼
      Calibration
           │
           ▼
       Evaluation
           │
           ▼
     API deployment
           │
           ▼
Inspection prioritization
           │
           ▼
     Field results
           │
           ▼
    Label ingestion
           │
           ▼
    Active learning
           │
           └──────► retraining
```
With monitoring around every transition.

## 🔍 55. Production Monitoring

A future deployment should monitor at least:

**Data drift**
- Feature distributions
- Missingness
- Categorical frequencies
- Geographic distribution

**Prediction drift**
- P(Lead) distribution
- Priority-score distribution
- Uncertainty distribution

**Label drift**
When new inspection results arrive:
- Observed lead rate
- Precision
- Recall
- PR-AUC
- Calibration

**Fairness drift**
- Inspection allocation
- False-negative disparities
- Coverage disparities
- Neighborhood representation

## 🛠️ 56. Troubleshooting

**`features_sample.parquet` does not exist**
Run:
```bash
python -m leadguard.data.clean \
  --input data/sample \
  --output data/interim/sample_interim.parquet
```
then:
```bash
python -m leadguard.data.features \
  --input data/interim/sample_interim.parquet \
  --output data/processed/features_sample.parquet
```

**Tests fail on coverage**
Run:
```bash
python -m pytest \
  tests/ \
  --cov=src \
  --cov-report=term-missing
```
Inspect uncovered lines before adding tests merely to satisfy the percentage.

**Active learning is slow**
Use the fast mode where available:
```bash
python -m leadguard.models.active_learning \
  --fast \
  --features data/processed/features_sample.parquet
```
The fast mode is intended for development/testing. It should not silently become the benchmark configuration.

## 🧠 57. Engineering Philosophy

LeadGuard follows several principles.

**Principle 1 — Evaluation is part of the model**
A model and its evaluation protocol cannot be treated independently.

**Principle 2 — Information provenance matters**
Every label-dependent feature should have a traceable source.

**Principle 3 — Uncertainty is a first-class output**
A model should be allowed to say: *"I don't know."*

**Principle 4 — Probability is not certainty**
A calibrated 0.87 is not a guarantee.

**Principle 5 — Fairness is a systems problem**
It cannot always be solved by putting protected variables into the model.

**Principle 6 — Active learning changes the data-generating process**
Once inspections generate new labels, the model's information state changes.

**Principle 7 — Explainability should support decisions**
SHAP is not included because colorful plots look impressive. It exists so a human can investigate why a property was prioritized.

**Principle 8 — Privacy is part of ML engineering**
A correct prediction does not justify exposing unnecessary personal/location information.

**Principle 9 — A failed experiment can be a successful engineering outcome**
Discovering leakage and reducing an inflated metric is not regression. It is improved scientific validity.

## 🏆 58. Why This Project Is Interesting

LeadGuard is deliberately more than:
`CSV ↓ XGBoost ↓ accuracy`

It combines:
```text
                         LEADGUARD
                             │
      ┌──────────────────────┼───────────────────────┐
      │                      │                       │
      ▼                      ▼                       ▼
   Spatial              Uncertainty                Equity
      ML                     │                       │
      │                      │                       │
      ▼                      ▼                       ▼
 Leakage-safe            Conformal                Priority
   features              prediction               scoring
      │                      │                       │
      └────────────┬─────────┴───────────────────────┘
                   ▼
            Active Learning
                   │
                   ▼
            New inspections
                   │
                   ▼
             New knowledge
                   │
                   ▼
              Better model
```

The hard part is not fitting XGBoost.
The hard part is making sure the entire system remains logically correct when:
- geography matters,
- labels arrive over time,
- uncertainty matters,
- fairness matters,
- privacy matters,
- and evaluation itself can leak information.

## 🧪 59. Reproducible Quality Gate

Before merging a substantial change, run:
```bash
python -m ruff check .
python -m ruff format --check .
python -m pytest \
  tests/ \
  --cov=src \
  --cov-report=term-missing \
  --cov-fail-under=80
```

Then inspect:
1. test result
2. coverage
3. warnings
4. metric changes
5. leakage tests
6. API schema changes

A green test suite is necessary.
It is not sufficient.

## 📜 60. Evaluation Contract

Every future headline metric should answer five questions:

1. **What dataset?** Synthetic sample? Real data? External validation set?
2. **What split?** Random? Geographic? Temporal?
3. **What information was available?** Especially for spatial label-derived features.
4. **Was the test set touched?** The answer should be: No.
5. **What does the number actually mean?**
   For example: *"0.41 PR-AUC on the bundled synthetic geographic-holdout evaluation"* is scientifically meaningful. *"LeadGuard achieves 0.41 PR-AUC"* is ambiguous.

## 🚨 61. Before vs After

**BEFORE**
```text
Headline metric
       │
       ▼
 PR-AUC ≈ 1.0
       │
       ▼
Looks impressive
       │
       ▼
  Deeper audit
       │
       ▼
Spatial label leakage
       │
       ▼
Evaluation invalid
```

**AFTER**
```text
  Split first
       │
       ▼
Reference-only spatial labels
       │
       ▼
Train / calibration / test
       │
       ▼
Calibrated probability
       │
       ▼
Binary conformal uncertainty
       │
       ▼
True active-learning retraining
       │
       ▼
Equity-aware prioritization
       │
       ▼
Leakage regression tests
       │
       ▼
  Honest evaluation
       │
       ▼
 Geo PR-AUC ≈ 0.41
```

## 💡 62. The Main Lesson

If there is only one thing a reviewer should remember about LeadGuard, it should be this:

**The goal was not to make the metric look good. The goal was to make the evaluation impossible to fool accidentally.**

That changed the architecture.
It changed the feature pipeline.
It changed the active-learning loop.
It changed the uncertainty implementation.
It changed the tests.
And it changed the reported performance.

That is exactly what a serious ML engineering project should do when an evaluation flaw is discovered.

## 📌 63. Limitations

LeadGuard currently has important limitations.

**Synthetic sample data**
The bundled sample is synthetic. Therefore: `sample PR-AUC ≠ real-world PR-AUC`
The data card explicitly identifies this limitation. ([GitHub](https://github.com/HarshkumarG007/LeadGuard/blob/main/docs/data_card.md))

**Geographic scope**
The production-oriented design is focused on the Chicago/Cook County context.

**Label quality**
Real service-line inventories may contain:
- self-reported labels,
- outdated records,
- incorrect material classifications.

**Spatial stationarity**
The model assumes that spatial relationships learned from historical data remain useful enough for the deployment environment. That assumption requires monitoring.

**Calibration**
Calibration must be continuously re-evaluated after:
- retraining,
- dataset shift,
- geographic expansion,
- policy changes.

**Fairness**
An equity-aware priority score does not mathematically guarantee equal outcomes. It is a mechanism for explicitly incorporating equity into decision-making.

## 🔮 64. Future Research

Several directions would substantially strengthen LeadGuard.

**Temporal leakage prevention**
Instead of only `train / test`, use `past → future` so the model is evaluated as it would actually operate over time.

**Spatial-temporal validation**
Combine `geographic holdout + temporal holdout` for a harder generalization test.

**Bootstrap confidence intervals**
Report `PR-AUC = 0.41, 95% CI = [...]` rather than a single point estimate.

**Cost-aware active learning**
Optimize $\frac{\text{Expected Value}}{\text{Inspection Cost}}$ rather than ranking purely by model properties.

**Human-in-the-loop learning**
Capture inspector feedback:
`prediction ↓ inspection ↓ actual material ↓ inspector notes ↓ new training signal`

**Monitoring**
Introduce automated detection of feature drift, geographic drift, calibration drift, fairness drift, and label-quality drift.

## 🤝 65. Contributing

Contributions should preserve the project's core invariants.
Before submitting a change:
```bash
python -m ruff check .
python -m ruff format --check .
python -m pytest tests/ --cov=src --cov-fail-under=80
```
For changes affecting spatial features, also add or update leakage tests.
For changes affecting active learning, verify that:
`new labels ↓ feature rebuild ↓ model retraining`
still occurs in that order.

## 📄 66. License

See the repository license file for the applicable terms.

## 👤 67. Author

**Harsh Kumar G**
LeadGuard is developed as an ML engineering/research portfolio project exploring:
- spatial machine learning,
- uncertainty quantification,
- active learning,
- fairness-aware decision systems,
- explainable AI,
- ML evaluation integrity,
- privacy-conscious serving.

## ⭐ 68. Final Takeaway

LeadGuard started with a seemingly simple goal:
*Find the properties most likely to have lead service lines.*

The deeper problem turned out to be much harder:
*How do you build a system that remains trustworthy when spatial correlation, sparse labels, expensive inspections, uncertainty, fairness, privacy, and evaluation leakage all interact?*

The answer implemented here is a complete decision pipeline:
```text
          DATA
           │
           ▼
        CLEANING
           │
           ▼
  SPLIT-FIRST FEATURES
           │
           ▼
        XGBOOST
           │
           ▼
      CALIBRATION
           │
    ┌──────┴───────┐
    ▼              ▼
CONFORMAL        SHAP
UNCERTAINTY   EXPLANATION
    │              │
    └──────┬───────┘
           ▼
  RISK + UNCERTAINTY
           │
           ▼
         EQUITY
           │
           ▼
  INSPECTION PRIORITY
           │
           ▼
      FIELD LABEL
           │
           ▼
    ACTIVE LEARNING
           │
           └──────────► MODEL UPDATE
```

And the project's most important evaluation result is not simply `PR-AUC = 0.41`.
It is the engineering transformation:

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

LeadGuard is therefore a demonstration of a broader ML engineering principle: trustworthy machine learning is not just about building a powerful model. It is about controlling information flow, measuring uncertainty, validating assumptions, protecting users, and being willing to report a lower number when the lower number is the truth.
