# 🛡️ LeadGuard
Uncertainty-Aware Lead Service Line Risk Intelligence

LeadGuard is an end-to-end machine-learning decision-support system for identifying properties that may contain lead service lines, quantifying prediction uncertainty, allocating limited inspection budgets, learning from newly inspected properties, and monitoring whether inspection decisions remain equitable across communities.

**Important:** LeadGuard is a decision-support and inspection-prioritization system. It does not certify that a property contains or does not contain a lead service line. Final determination requires physical inspection or an authoritative record.

## 🥭 What Does "LeadGuard" Actually Do?
Imagine a city has:
- 100,000 properties
- only $500,000 available for inspections
- limited inspectors
- incomplete service-line records
- uncertain historical data
- neighborhoods with very different levels of documentation

The city cannot inspect everything.

The real question is therefore not simply:
> "Which properties are likely to have lead?"

It is:
> "Given incomplete information, uncertainty, limited money, and fairness constraints, which properties should we inspect first?"

LeadGuard is designed to answer that question.

At a high level:

```
    ┌──────────────────────┐
    │ Property / Inventory │
    │         Data         │
    └──────────┬───────────┘
               │
               ▼
    ┌──────────────────────┐
    │ Leakage-Safe Feature │
    │     Engineering      │
    └──────────┬───────────┘
               │
               ▼
    ┌──────────────────────┐
    │   Lead Risk Model    │
    │       XGBoost        │
    └──────────┬───────────┘
               │
               ▼
    ┌──────────────────────┐
    │     Probability      │
    │     Calibration      │
    └──────────┬───────────┘
               │
     ┌─────────┴─────────┐
     │                   │
     ▼                   ▼
┌──────────────────┐ ┌──────────────────┐
│    Predictive    │ │    Conformal     │
│   Uncertainty    │ │    Prediction    │
└────────┬─────────┘ └────────┬─────────┘
         │                    │
         └─────────┬──────────┘
                   │
                   ▼
    ┌──────────────────────┐
    │  Equity / Coverage   │
    │      Controller      │
    └──────────┬───────────┘
               │
               ▼
    ┌──────────────────────┐
    │ Inspection Priority  │
    │  / Budget Optimizer  │
    └──────────┬───────────┘
               │
               ▼
        HUMAN INSPECTION
               │
               ▼
       New Ground Truth
               │
               ▼
    ┌──────────────────────┐
    │   Active Learning    │
    │     / Retraining     │
    └──────────┬───────────┘
               │
               └──────► repeat
```

The important idea is that the ML model is only one component.
LeadGuard is fundamentally a sequential decision-making system under uncertainty.

## 📌 Table of Contents

- [Why LeadGuard Exists](#why-leadguard-exists)
- [The Problem](#the-problem)
- [The Core Insight](#the-core-insight)
- [What LeadGuard Predicts](#what-leadguard-predicts)
- [How the System Works](#how-the-system-works)
- [End-to-End Pipeline](#end-to-end-pipeline)
- [Data](#data)
- [Feature Engineering](#feature-engineering)
- [The Most Important Design Rule: No Label Leakage](#the-most-important-design-rule-no-label-leakage)
- [Train / Calibration / Test Architecture](#train--calibration--test-architecture)
- [Machine Learning Model](#machine-learning-model)
- [Probability Calibration](#probability-calibration)
- [Uncertainty Quantification](#uncertainty-quantification)
- [Conformal Prediction](#conformal-prediction)
- [Inspection Prioritization](#inspection-prioritization)
- [Equity](#equity)
- [Active Learning](#active-learning)
- [Active Learning Strategies](#active-learning-strategies)
- [Budget-Constrained Decision Making](#budget-constrained-decision-making)
- [Explainability](#explainability)
- [API](#api)
- [Privacy](#privacy)
- [Streamlit Dashboard](#streamlit-dashboard)
- [Project Architecture](#project-architecture)
- [Repository Structure](#repository-structure)
- [Configuration](#configuration)
- [Installation](#installation)
- [Running LeadGuard](#running-leadguard)
- [Running Tests](#running-tests)
- [ML Evaluation](#ml-evaluation)
- [Leakage Testing](#leakage-testing)
- [Calibration Evaluation](#calibration-evaluation)
- [Active Learning Evaluation](#active-learning-evaluation)
- [Reproducibility](#reproducibility)
- [Engineering Principles](#engineering-principles)
- [Limitations](#limitations)
- [What LeadGuard Does Not Claim](#what-leadguard-does-not-claim)
- [Roadmap](#roadmap)
- [Research Questions](#research-questions)
- [Why This Architecture Matters](#why-this-architecture-matters)
- [Contributing](#contributing)
- [License](#license)

## 🧠 Why LeadGuard Exists

Lead service lines are a public-health and infrastructure problem.

A city may know that some properties contain lead service lines, but often does not know the status of every property.

Historical records can be incomplete.

Some properties may have:
- missing records
- outdated records
- inconsistent material classifications
- uncertain installation dates
- incomplete geographic information
- self-reported information
- neighborhood-level correlations

At the same time, physical inspections cost money and require human resources.

This creates a constrained decision problem:

```
        LIMITED INSPECTION BUDGET
                    │
                    ▼
       ┌─────────────────────────┐
       │ Which properties should │
       │   be inspected first?   │
       └─────────────────────────┘
          ▲         ▲         ▲
          │         │         │
        Risk   Uncertainty  Equity
```

LeadGuard is designed around this problem.

## 🎯 The Problem

A conventional ML project might formulate the task as:
`Input: property features` → `Output: Lead / Not Lead`

LeadGuard goes further.

The actual decision pipeline is:
- Property
- ↓ Estimated probability of Lead
- ↓ How trustworthy is that probability?
- ↓ How uncertain is the prediction?
- ↓ How much should this property be prioritized?
- ↓ Does the resulting inspection allocation remain equitable?
- ↓ What should we inspect next?
- ↓ What did we learn?
- ↓ How should the model update?

This changes the project from a simple classification problem into an uncertainty-aware resource allocation system.

## 💡 The Core Insight

The most important conceptual distinction in LeadGuard is:
**Prediction is not the same thing as decision-making.**

A model might say:
- Property A → 92% Lead risk
- Property B → 89% Lead risk

But suppose:
- A is in a heavily inspected neighborhood.
- B is in a neighborhood that has historically received very few inspections.
- B's prediction is highly uncertain.
- B's inspection could provide valuable information for future predictions.

Then simply sorting by probability may not be the best policy.

LeadGuard therefore considers several dimensions:

```
       ┌───────────────┐
       │   Lead Risk   │
       └───────┬───────┘
               │
               ▼
       ┌───────────────┐
       │  Uncertainty  │
       └───────┬───────┘
               │
               ▼
       ┌───────────────┐
       │    Equity     │
       └───────┬───────┘
               │
               ▼
       ┌───────────────┐
       │    Budget     │
       └───────┬───────┘
               │
               ▼
        INSPECTION QUEUE
```

## 🔬 What LeadGuard Predicts

The core supervised-learning target is intentionally binary:
- `y = 1` → Lead
- `y = 0` → NotLead

The primary model estimates:
`P(Lead | property information)`

For example:
`Property ID: 10482` → `P(Lead) = 0.82`

This means:
*Based on the information available to the model, the estimated probability of a Lead service line is 82%.*

It does **not** mean:
*There is an 82% laboratory-certified chance that the pipe is lead.*

The distinction matters.
Predictions are estimates conditioned on the available data and model assumptions.

## 🏗️ How the System Works

LeadGuard consists of several logical layers.

### 1. Data Layer
Contains:
- property information
- service-line information
- geographic information
- historical records
- socioeconomic/reference information used for auditing
- inspection outcomes

### 2. Feature Layer
Transforms raw records into predictive features.
Examples include:
- construction year
- property characteristics
- geographic information
- neighborhood statistics
- spatial proximity
- historical local Lead prevalence

**Label-dependent spatial features are constructed only from allowed reference data.**

### 3. Model Layer
The main predictive model is:
**XGBoost**

It produces:
`P(Lead)`

### 4. Calibration Layer
Raw tree-model probabilities are not automatically reliable probabilities.
LeadGuard therefore applies probability calibration using a held-out calibration set.

The objective is to make:
`predicted probability ≈ observed frequency`

For example, among properties assigned:
`P(Lead) ≈ 0.80`
we would ideally observe Lead in approximately:
`80%`
of such cases over a sufficiently large sample.

### 5. Uncertainty Layer
LeadGuard separately estimates predictive uncertainty.

A probability near:
`0.50`
is inherently more ambiguous than:
`0.99`

The system therefore exposes uncertainty independently from probability.

### 6. Conformal Layer
Conformal prediction provides prediction sets with a target coverage guarantee under its assumptions.

For binary Lead prediction, a property may receive:
`["Lead"]`
or:
`["NotLead"]`
or:
`["NotLead", "Lead"]`

The final case means:
*The available evidence is insufficient to confidently exclude either class at the selected conformal level.*

### 7. Decision Layer
Risk, uncertainty, equity, and budget constraints are combined into an inspection-prioritization policy.

### 8. Active Learning Layer
When a human inspector provides a new verified label:
`Property → inspected → actual result`
the new information becomes training data.
The model can then be retrained.

This creates a feedback loop:
`Predict ↓ Prioritize ↓ Inspect ↓ Observe ↓ Learn ↓ Predict again`

## 🔄 End-to-End Pipeline

The complete lifecycle is:

```
        ┌─────────────────────┐
        │      Raw Data       │
        └──────────┬──────────┘
                   │
                   ▼
        ┌─────────────────────┐
        │   Data Validation   │
        └──────────┬──────────┘
                   │
                   ▼
        ┌─────────────────────┐
        │    Define Splits    │
        │ Train / Cal / Test  │
        └──────────┬──────────┘
                   │
                   ▼
     ┌────────────────────────────┐
     │ Leakage-Safe Feature Build │
     │ using TRAIN reference only │
     └────────────┬───────────────┘
                   │
                   ▼
           ┌──────────────┐
           │   XGBoost    │
           └──────┬───────┘
                   │
                   ▼
           ┌──────────────┐
           │ Calibration  │
           │ on CAL only  │
           └──────┬───────┘
                   │
                   ▼
           ┌──────────────┐
           │     TEST     │
           │  Evaluation  │
           └──────┬───────┘
                  │
        ┌─────────┴────────┐
        ▼                  ▼
   Probability        Uncertainty
        │                  │
        └─────────┬────────┘
                  ▼
           Conformal Layer
                  │
                  ▼
             Equity Layer
                  │
                  ▼
            Priority Queue
                  │
                  ▼
           Human Inspection
                  │
                  ▼
              New Labels
                  │
                  ▼
           Active Learning
                  │
                  └──────────────► Retrain
```

## 🗃️ Data

LeadGuard is designed to work with property/service-line inventory data.

The repository also contains a synthetic dataset for reproducible development and testing.

### Synthetic data
The synthetic dataset exists to:
- exercise the pipeline
- test feature engineering
- test model training
- test uncertainty logic
- test active learning
- test API behavior
- make CI reproducible

Synthetic metrics should not be interpreted as evidence of real-world Chicago performance.

## 🧱 Feature Engineering

LeadGuard distinguishes between two kinds of features.

### Label-independent features
These can be calculated without knowing Lead labels.
Examples:
- year_built
- property characteristics
- geographic coordinates
- non-target structural attributes

These are safe to compute before splitting.

### Label-dependent features
These use known Lead outcomes.
Examples:
- neighbor_lead_rate
- knn_lead_rate
- dist_to_nearest_known_lead_m

These are fundamentally different.
**They must only use labels available in the permitted training/reference population.**

## 🚨 The Most Important Design Rule: No Label Leakage

The single most important ML-evaluation rule in LeadGuard is:
**Information from the evaluation set must never be allowed to influence its own features, model, calibration, or decisions.**

Consider:
```
TEST PROPERTY
 ├── actual label = Lead
 └── feature: distance to nearest known Lead
```

If the nearest-known-Lead calculation includes the test property itself, the model can effectively see the answer.
That is leakage.

### 🔐 Split-First Feature Architecture

LeadGuard therefore follows:

```
        RAW DATA
           │
           ▼
      DEFINE SPLIT
           │
     ┌─────┴────────┐
     │              │
   TRAIN          TEST
     │              │
     ▼              │
 Reference          │
  labels            │
     │              │
     └──────┬───────┘
            ▼
    Build TEST features
 using TRAIN reference only
```

The forbidden architecture is:
`RAW DATA ↓ BUILD ALL FEATURES ↓ SPLIT`
because label-dependent features may already contain information from the eventual test set.

### 🧪 Leakage Invariants

LeadGuard includes explicit tests intended to catch this class of bug.

#### Test 1 — Test-label invariance
Changing TEST labels must not change:
- TEST features
- TRAIN features
- trained model
- calibration model
- predictions

Only evaluation metrics may change.

#### Test 2 — Training-label permutation
Randomizing training labels should destroy meaningful predictive performance.
The resulting PR-AUC should approach the prevalence baseline rather than remaining suspiciously high.
This acts as a practical leakage detector.

#### Test 3 — Geographic isolation
Label-derived geographic features for the test partition must be computed from permitted training/reference data only.

## 🧩 Train / Calibration / Test Architecture

LeadGuard uses a strict three-way evaluation concept:

```
    ┌──────────────────────────┐
    │       TRAIN — 70%        │
    │                          │
    │       Fit XGBoost        │
    └────────────┬─────────────┘
                 │
                 │
    ┌────────────▼─────────────┐
    │    CALIBRATION — 15%     │
    │                          │
    │     Fit probability      │
    │        calibrator        │
    └────────────┬─────────────┘
                 │
                 │
    ┌────────────▼─────────────┐
    │        TEST — 15%        │
    │                          │
    │  Final evaluation only   │
    └──────────────────────────┘
```

The test set is treated as locked until final evaluation.

## 🤖 Machine Learning Model

The primary model is XGBoost.

### Why XGBoost?
Because LeadGuard deals with structured/tabular data where:
- nonlinear relationships matter
- feature interactions matter
- missingness may be informative
- mixed feature scales are common
- strong tabular performance is desirable
- interpretability tools such as SHAP are available

The model learns:
`X → P(Lead)`
where X is the leakage-safe feature vector.

### 📏 Model Evaluation

LeadGuard should not rely on a single metric.
Relevant metrics include:

#### PR-AUC
Precision-recall area under the curve is especially useful when Lead properties are relatively uncommon.
It evaluates ranking quality for the positive class.

#### ROC-AUC
Useful as a general ranking metric, but less informative than PR-AUC when the positive class is highly imbalanced.

#### Precision
Of properties predicted as Lead:
How many were actually Lead?

#### Recall
Of actual Lead properties:
How many did we identify?

#### FNR
False-negative rate:
`FN / (FN + TP)`
This is particularly important in a public-health inspection context because missing a genuinely Lead property may be more consequential than inspecting a non-Lead property.

## 🎯 Probability Calibration

A model score is not automatically a trustworthy probability.

For example:
`Raw model: 0.90`
does not necessarily mean:
`90% of similar properties are actually Lead.`

Calibration attempts to make that interpretation more defensible.

LeadGuard uses a held-out calibration set:
`TRAIN ↓ XGBoost ↓ raw probabilities`
`CALIBRATION ↓ fit sigmoid/Platt calibrator`
`TEST ↓ calibrated probabilities`

### 📊 Calibration Metrics

LeadGuard evaluates calibration using metrics such as:

#### Brier Score
Measures squared probability error.
Lower is better.

#### Log Loss
Penalizes incorrect predictions, particularly confident incorrect predictions.
Lower is better.

#### Expected Calibration Error
ECE compares predicted confidence with empirical accuracy across probability bins.
Conceptually:
`Predicted 0.8 ↓ Observed frequency should be near 0.8`

#### Reliability Curve
A visual comparison between:
`predicted probability`
and:
`observed frequency`

## 🎲 Uncertainty Quantification

LeadGuard intentionally separates three concepts:
- P(Lead)
- Predictive uncertainty
- Conformal prediction set

They are related, but they are not identical.

### 🌡️ Predictive Uncertainty

For a binary probability p, LeadGuard can use:
`uncertainty = 1 - |2p - 1|`

Interpretation:
- `p = 0.50 → uncertainty = 1.00`
- `p = 0.75 → uncertainty = 0.50`
- `p = 0.99 → uncertainty = 0.02`

Thus:
- `0` → very confident
- `1` → maximally ambiguous

This quantity is useful for ranking properties where the model is least decisive.

### 🧮 Why Not Use the Old "Fake Ensemble"?

An earlier experimental implementation perturbed input features with Gaussian noise and treated the resulting variation as ensemble disagreement.

That is not a genuine model ensemble.

A true ensemble would involve multiple independently trained models, for example:
`Model 1`
`Model 2`
`Model 3`
`Model 4`
`Model 5`
`↓ distribution of predictions`

LeadGuard deliberately removes the pseudo-ensemble approach rather than giving a misleading interpretation to feature perturbation.
A genuine epistemic ensemble can be added later.

## 🧾 Conformal Prediction

LeadGuard uses binary conformal semantics:
- `0` → NotLead
- `1` → Lead

Possible output:
`{ "conformal_set": ["Lead"] }`
or:
`{ "conformal_set": ["NotLead", "Lead"] }`

The second result means the model cannot confidently exclude either class at the selected conformal level.

### 🛡️ Why Conformal Prediction Matters

A conventional classifier might say:
`Lead probability = 0.58`

A decision-maker may reasonably ask:
"How certain are we?"

Conformal prediction gives an additional statistical framework for controlling coverage under its assumptions.
The goal is not simply:
`maximum accuracy`
but:
`valid uncertainty with useful prediction sets`

### 📐 Conformal Evaluation

LeadGuard evaluates more than coverage.
Important measurements include:
- empirical coverage
- average prediction-set size
- singleton-set rate
- coverage across risk groups
- coverage across geographic groups
- coverage across probability/risk bins

A trivial system that returns:
`["NotLead", "Lead"]`
for every property could achieve high coverage while being practically useless.

Therefore:
Coverage must be considered together with prediction-set efficiency.

## 🚦 Inspection Prioritization

The ultimate purpose of LeadGuard is not classification.
It is prioritization.

The system produces a ranked inspection queue.
Conceptually:
`Property ↓ Risk + Uncertainty + Equity ↓ Priority`

A simplified conceptual priority function is:
`Priority = Risk Component + Uncertainty Component + Equity Component`

The exact weights are configurable.

## ⚖️ Equity

A purely risk-maximizing strategy can repeatedly select properties from already well-documented areas.

That can create a feedback loop:
`More inspections ↓ More labels ↓ Better model confidence ↓ More prioritization ↓ Even more inspections`

Meanwhile, under-inspected communities can remain data-poor.

LeadGuard therefore treats equity as an explicit decision-layer concern.

### 🧭 Equity Is Not the Same as Demographic Prediction

LeadGuard separates predictive modeling from fairness auditing.

Sensitive or socioeconomic variables should not automatically become predictive features simply because they correlate with the target.

Instead, such variables can be used for:
- evaluation
- auditing
- allocation monitoring
- disparity analysis

This separation helps answer:
"Is the system behaving equitably?"
without automatically asking:
"Can the model exploit this demographic information to improve prediction?"

### 📊 Fairness Metrics

LeadGuard can evaluate disparities such as:

#### False-negative-rate disparity
Compare:
`FNR(group A) vs FNR(group B)`

#### Inspection allocation
Compare:
`inspection share vs risk share`

#### Discovery efficiency
Measure:
`Lead discoveries / inspections`
by group.

#### Inspection delay
Measure how long different communities are expected to wait before inspection.

This is especially important because LeadGuard makes resource allocation decisions, not merely classification decisions.

## 🔁 Active Learning

The active-learning loop is one of LeadGuard's most important concepts.

Initially:
Only some properties are labeled.
LeadGuard predicts on the unknown pool.

Then:
Select properties for inspection.
A human inspector provides the actual result.
That new information becomes labeled data.

Then:
Retrain.
The process repeats.

### 🧠 Active Learning Loop

```
       ┌──────────────────┐
       │ Initial Labels L₀│
       └────────┬─────────┘
                ▼
      Build Features using L₀
                │
                ▼
           Train Model
                │
                ▼
        Score Unknown Pool
                │
                ▼
        Choose Inspections
                │
                ▼
         Human Inspection
                │
                ▼
            New Labels
                │
                ▼
         L₁ = L₀ + new
                │
                ▼
      Rebuild Features using L₁
                │
                ▼
          Retrain Model
                │
                └──────► repeat
```

The important detail is:
**Label-dependent spatial features are rebuilt after every acquisition round.**
Otherwise the model would not actually learn from newly acquired spatial information.

### 🧪 Active Learning Strategies

LeadGuard compares multiple acquisition policies.

#### 1. Random
Select properties randomly.
This is the baseline.

#### 2. Highest Risk
Inspect properties with the highest:
`P(Lead)`
This answers:
What happens if we only care about immediate risk?

#### 3. Highest Uncertainty
Inspect properties where the model is least certain.
This answers:
What happens if we prioritize information gain?

#### 4. Risk + Uncertainty
Balance:
high expected risk
with:
high uncertainty

#### 5. Risk + Uncertainty + Equity
Add the allocation/equity component.
This tests whether a constrained strategy can preserve useful discovery performance while improving coverage.

#### 6. Oracle
The oracle has access to ground truth for simulation purposes.
It represents an upper-bound reference rather than a deployable strategy.
The oracle answers:
"How good could the acquisition process be if we already knew the true labels?"

### 📈 What Active Learning Should Measure

For every strategy, LeadGuard can track:
`Number of inspections ↓ True Lead discoveries ↓ Cumulative discoveries`

Additional useful metrics include:
- Lead discoveries / inspection
- Lead discoveries / dollar spent
- Coverage across communities
- Average uncertainty reduction

The goal is not simply to produce a better classifier.
The goal is:
**Get more useful information from the same inspection budget.**

## 💰 Budget-Constrained Decision Making

Suppose:
- Budget = $500,000
- Inspection cost = $500

Then:
Maximum inspections = 1,000

LeadGuard can rank candidate properties and allocate the available inspection budget.

For variable inspection costs, the underlying decision problem becomes closer to a constrained optimization problem:
`maximize expected value`
`subject to: total inspection cost ≤ budget`

Future versions can extend this into a formal knapsack/resource-allocation optimizer.

## 🔍 Explainability

A high-risk prediction should not simply appear as:
`P(Lead) = 0.91`

Decision-makers should be able to ask:
**Why?**

LeadGuard uses tree-model explainability techniques such as SHAP to identify influential features.

A conceptual explanation might look like:
```
Lead Risk: 91%

Factors increasing risk:
+ Older construction year
+ High local Lead prevalence
+ Spatial proximity to known Lead properties

Factors decreasing risk:
- More recent construction
- Low-risk property characteristics
```

Explainability is intended to support human review.
It should not be interpreted as causal inference.

## 🌐 API

LeadGuard exposes an API layer for integrating the model with applications.

The API can provide information such as:
- property risk
- probability of Lead
- uncertainty
- conformal prediction
- priority
- explanation information
- inspection queue

### 🔒 API Privacy

The public priority queue intentionally does not expose property addresses.

The public-facing response should use identifiers such as:
`property_id`
and other non-address identifiers where appropriate.

The principle is:
`Public analytics ↓ minimal identifying information`
`Authorized operational workflow ↓ property lookup ↓ address`

This prevents a ranking endpoint from becoming an unnecessary address-disclosure mechanism.

## 🖥️ Streamlit Dashboard

LeadGuard includes a Streamlit-based interface for exploring the system.

The dashboard is intended to make the model understandable to humans rather than forcing users to interact directly with raw API responses.

Typical workflow:
`Load data ↓ Explore properties ↓ Inspect model risk ↓ Understand uncertainty ↓ Review priority ↓ Explore geographic patterns ↓ Review inspection allocation`

The dashboard should consume the privacy-preserving API representation and use property_id rather than assuming an address is present in the public queue response.

## 🏛️ Project Architecture

```text
LeadGuard/
│
├── app/
│   └── streamlit_app.py
│
├── api/
│   └── main.py
│
├── src/
│   └── leadguard/
│       │
│       ├── data/
│       │   ├── features.py
│       │   └── ...
│       │
│       ├── models/
│       │   ├── xgboost_model.py
│       │   ├── baseline.py
│       │   ├── uncertainty.py
│       │   └── active_learning.py
│       │
│       ├── evaluation/
│       │   ├── fairness.py
│       │   └── ...
│       │
│       └── ...
│
├── tests/
│   ├── unit/
│   │   ├── test_features.py
│   │   ├── test_uncertainty.py
│   │   ├── test_leakage.py
│   │   └── test_calibration.py
│   │
│   └── integration/
│       └── test_privacy.py
│
├── configs/
├── models/
│   └── xgb_model.pkl
├── data/
├── docs/
│   ├── LeadGuard_Architecture.md
├── PROGRESS.md
├── pyproject.toml
└── README.md
```

The exact contents may evolve, but the architectural separation is intentional.

### 📁 Repository Structure

**src/leadguard/data/**
Responsible for:
- loading data
- validation
- feature engineering
- split-aware spatial features

**src/leadguard/models/**
Contains the predictive and learning components:
- XGBoost
- baselines
- uncertainty
- active learning

**src/leadguard/evaluation/**
Contains evaluation logic such as:
- fairness
- calibration
- model metrics
- decision-level analysis

**api/**
Contains the service layer.

**app/**
Contains the interactive dashboard.

**tests/**
Contains:
- unit tests
- integration tests
- leakage tests
- privacy tests
- calibration tests

## ⚙️ Configuration

LeadGuard is designed to keep important parameters configurable rather than hard-coded.

Typical configuration categories include:
- data paths
- model hyperparameters
- random seeds
- train/calibration/test proportions
- inspection costs
- budget
- uncertainty parameters
- equity weights
- active-learning batch size
- number of active-learning rounds

This makes experiments reproducible and allows different decision policies to be compared without rewriting core code.

## 🚀 Installation

Clone the repository:
```bash
git clone https://github.com/HarshkumarG007/LeadGuard.git
cd LeadGuard
```

Create and activate a virtual environment:
```bash
python -m venv .venv
```
Windows:
```bash
.venv\Scripts\activate
```
macOS / Linux:
```bash
source .venv/bin/activate
```

Install the project dependencies according to the repository's dependency configuration.
For example:
```bash
pip install -e .
```

## ▶️ Running LeadGuard

The exact commands may depend on the configured environment and repository scripts.
Typical components include:

**Train / evaluate model**
```bash
python -m leadguard.models.xgboost_model
```
This should:
- load validated data
- define evaluation splits
- build leakage-safe features
- train XGBoost
- calibrate probabilities
- evaluate on the held-out test set
- save the calibrated model

**Run tests**
```bash
pytest
```

**Run the API**
A typical FastAPI development invocation is:
```bash
uvicorn api.main:app --reload
```

**Run Streamlit**
```bash
streamlit run app/streamlit_app.py
```

## 🧪 Running Tests

The test suite is an important part of LeadGuard because many of the most dangerous ML failures do not produce Python exceptions.

A model can:
run successfully
and still be scientifically invalid.

Therefore LeadGuard tests both:
- software correctness
and:
- evaluation correctness

### 🚨 Leakage Testing
The leakage test suite is particularly important.
It checks invariants such as:

**Test-label invariance**
Changing test labels must not alter test features.

**Training-label permutation**
Randomized training labels should destroy meaningful predictive signal.

**Spatial reference isolation**
Test geography must not contribute test labels to spatial aggregates.

**Feature provenance**
Label-dependent features must have an explicit reference population.

### 📏 Calibration Evaluation
Calibration tests should verify:
- probabilities are in [0, 1]
- calibration uses held-out data
- Brier score is calculable
- log loss is calculable
- ECE is calculable
- reliability curves can be generated
- test labels do not influence the calibrator

### 🔄 Active Learning Evaluation
The active-learning evaluation should compare:
- Random
- Highest Risk
- Highest Uncertainty
- Risk + Uncertainty
- Risk + Uncertainty + Equity
- Oracle

Each strategy is evaluated across increasing inspection budgets.

Example:
```
Budget │
       │ ─── Oracle
       │ ─────
       │ ───── Risk+Uncertainty
       │ ─────
       │ ───── Risk
       │ ─── Random
       └────────────────────────── Number Inspected
```

The actual curves should be generated from experiments rather than manually assumed.

## 🔬 Evaluation Philosophy

LeadGuard deliberately avoids treating:
**one impressive metric**
as proof that the system works.

A serious evaluation asks:

- **Can the model predict?**
  PR-AUC, ROC-AUC, Precision, Recall
- **Are the probabilities meaningful?**
  Brier, Log Loss, ECE, Reliability
- **Is uncertainty useful?**
  Conformal Coverage, Set Size, Uncertainty Ranking
- **Does it work spatially?**
  Geographic Holdout, Neighborhood Holdout
- **Does it survive label noise?**
  Noise Stress Tests
- **Does active learning help?**
  Discoveries / Inspection, Discoveries / $
- **Is the allocation equitable?**
  FNR disparity, Inspection allocation, Inspection delay, Discovery rate
- **Does the API respect privacy?**
  Address exclusion

### 🧪 The Pre-Fix Evaluation Lesson

An earlier version of LeadGuard produced extremely high synthetic performance.
That result was not accepted at face value.

Investigation identified a serious potential source of evaluation contamination:
**label-dependent spatial features**
were being generated before the final evaluation split.

For example, a feature such as:
`distance to nearest known Lead`
can become invalid if the "known Lead" reference population contains eventual test labels.

Therefore:
The old perfect metrics are not treated as valid evidence of model generalization.

The evaluation pipeline was redesigned around split-first feature generation.
This is an intentional part of LeadGuard's engineering story.

### 🧯 Why This Matters

A model can have:
`PR-AUC = 1.00`
and still be wrong.

If the feature pipeline accidentally allows:
`TEST LABEL ↓ TEST FEATURE ↓ MODEL ↓ TEST PREDICTION`
then the evaluation is circular.

The correct relationship is:
`TRAIN LABEL ↓ TRAIN-DERIVED REFERENCE FEATURES ↓ MODEL ↓ TEST FEATURES ↓ TEST PREDICTION`

That distinction is more important than the headline metric.

## 🔁 Reproducibility

LeadGuard aims to make experiments reproducible through:
- fixed random seeds
- deterministic data-processing paths where practical
- configuration files
- explicit dataset versions
- automated tests
- documented evaluation procedures
- separation of training/calibration/test data

Reproducibility is particularly important for active-learning experiments because acquisition order can influence future model behavior.

## 🧱 Engineering Principles

LeadGuard follows several principles.

**1. Split before label-dependent feature engineering**
Never construct target-informed features on the entire dataset before evaluation splitting.

**2. Test data is sacred**
The test set should be touched only for final evaluation.

**3. Probability is not confidence**
A probability estimate and uncertainty estimate answer different questions.

**4. Calibration is part of the product**
If downstream decisions depend on probability magnitude, calibration is not optional decoration.

**5. Uncertainty should have a clear meaning**
Avoid inventing sophisticated terminology for heuristics.
If something is perturbation sensitivity, call it perturbation sensitivity.
If something is predictive entropy, call it predictive entropy.
If something is ensemble variance, it should come from an actual ensemble.

**6. Fairness must match the decision**
Because LeadGuard allocates inspections, fairness analysis should examine allocation and outcomes—not only classification statistics.

**7. Humans remain in the loop**
LeadGuard prioritizes inspections.
It does not replace physical inspection.

**8. Privacy by default**
Public endpoints should return only the information required for their purpose.

## ⚠️ Limitations

LeadGuard has important limitations.

**Synthetic data**
Synthetic data is useful for development but cannot establish real-world performance.

**Label quality**
Historical service-line records may contain:
- incorrect labels
- missing labels
- outdated records
- inconsistent definitions

A model cannot automatically correct bad ground truth.

**Geographic distribution shift**
A model trained in one geographic environment may not generalize to another.
Even nearby neighborhoods can have substantially different infrastructure histories.

**Spatial correlation**
Nearby properties can share infrastructure characteristics.
This is useful signal but also creates a major risk of leakage if spatial features are constructed incorrectly.

**Missing-not-at-random labels**
Properties with known service-line material may not be a random sample.
Inspection decisions themselves may have historically been targeted.
This creates selection bias.

**Calibration shift**
A calibrated model can become miscalibrated when the underlying population changes.
Calibration must therefore be monitored after deployment.

**Conformal assumptions**
Conformal prediction provides guarantees under assumptions about the data-generation process and exchangeability.
Real-world geographic and temporal distribution shifts can violate these assumptions.

**Equity trade-offs**
There is no universally correct fairness objective.
Increasing equity constraints may reduce immediate expected Lead discoveries.
That trade-off should be explicitly measured rather than hidden.

**Cost assumptions**
Inspection costs may vary.
Travel distance, property complexity, contractor availability, and administrative costs can make real-world costs substantially more complicated than a single fixed inspection price.

## 🚫 What LeadGuard Does Not Claim

LeadGuard does not claim:
- that every high-risk property contains Lead
- that every low-risk property is safe
- that ML predictions replace physical inspection
- that synthetic performance represents real-world performance
- that a calibrated probability is a guarantee
- that conformal prediction eliminates distribution shift
- that fairness can be reduced to one metric
- that historical labels are perfect
- that the current model is universally deployable
- that the system makes autonomous public-health decisions

The intended role is:
**Decision support for prioritizing limited inspection resources.**

## 🗺️ Roadmap

### Phase 1 — Evaluation integrity
- [x] Split-first architecture
- [x] Leakage-safe spatial features
- [x] Explicit leakage tests
- [x] Train/calibration/test separation
- [x] Binary uncertainty semantics
- [x] Remove pseudo-ensemble uncertainty
- [x] API address privacy

### Phase 2 — Stronger uncertainty
- [ ] Production-quality conformal calibration
- [ ] Conditional/group-wise coverage analysis
- [ ] Prediction-set efficiency analysis
- [ ] Genuine epistemic ensemble
- [ ] Drift-aware uncertainty monitoring

### Phase 3 — Active learning
- [ ] Iterative retraining architecture
- [ ] Rebuild spatial features after acquisition
- [ ] Strategy comparison
- [ ] Oracle benchmark
- [ ] More rigorous information-gain objectives
- [ ] Cost-aware acquisition

### Phase 4 — Decision optimization
- [ ] Formal budget optimizer
- [ ] Variable inspection costs
- [ ] Travel-distance constraints
- [ ] Contractor capacity
- [ ] Geographic coverage constraints
- [ ] Explicit fairness constraints

### Phase 5 — Real-world validation
- [ ] Real public dataset ingestion
- [ ] Data provenance documentation
- [ ] Temporal validation
- [ ] Geographic external validation
- [ ] Label-noise analysis
- [ ] Calibration monitoring
- [ ] Distribution-shift testing

## 🔬 Research Questions

LeadGuard can be viewed as an applied research platform for questions such as:

1. **How much does spatial information actually contribute?**
   Compare: non-spatial model vs spatial model
2. **Does uncertainty improve inspection efficiency?**
   Compare: risk-only vs risk + uncertainty using actual Lead discoveries per inspection.
3. **Does active learning outperform random inspection?**
   Measure: cumulative discoveries as inspection budget increases.
4. **What is the cost of equity constraints?**
   Compare: unconstrained optimization vs equity-constrained optimization and quantify the trade-off.
5. **How quickly does model performance improve as inspections accumulate?**
   Measure: number of labels ↓ model performance ↓ calibration ↓ uncertainty reduction
6. **How robust is the system to historical label bias?**
   Simulate: biased historical inspections and evaluate how quickly active learning can recover.

## 🧠 Why This Architecture Matters

LeadGuard is intentionally more than:
`CSV → XGBoost → accuracy`

The deeper architecture is:
```
       HIDDEN REALITY
             │
             ▼
       Partial labels
             │
             ▼
       ML probability
             │
             ▼
        Calibration
             │
             ▼
        Uncertainty
             │
             ▼
      Decision policy
             │
      ┌──────┴──────┐
      │             │
   Budget        Equity
      │             │
      └──────┬──────┘
             ▼
         Inspection
             │
             ▼
        New evidence
             │
             ▼
          Learning
             │
             └──────► better future decisions
```

This is why LeadGuard should be thought of as a:
**closed-loop uncertainty-aware inspection intelligence system**
rather than merely a Lead classification model.

## 🥭 Explained Like You're Five

Imagine a giant neighborhood where some houses might have old lead pipes.
You only have enough money to check 100 houses.

You ask a smart computer:
"Which houses should we check first?"

The computer looks at things such as:
- how old the house is
- what nearby houses look like
- what information is already known
- how uncertain the prediction is

Then it says:
- House A → probably Lead
- House B → probably not Lead
- House C → we're really unsure

But the computer shouldn't simply check the 100 houses it likes most.
It also asks:
"Have we been checking the same neighborhood over and over?"
and:
"Which inspection would teach us something useful?"

Then humans inspect selected houses.
The results go back into the system.
The computer learns.
Then it chooses the next houses.

So the loop is:
`GUESS ↓ CHECK ↓ LEARN ↓ GUESS BETTER ↓ CHECK ↓ LEARN ↓ ...`

That's LeadGuard.

## 🧠 Explained Like You're an ML Engineer

Formally, LeadGuard estimates:
`P(Y = Lead | X)`
using a gradient-boosted tree model.

The feature pipeline distinguishes:
`X_static`
from:
`X_label-dependent(reference = TRAIN)`
to prevent target leakage.

The evaluation architecture is:
`D = D_train ∪ D_cal ∪ D_test`
with:
`D_train ∩ D_cal = ∅`
`D_train ∩ D_test = ∅`
`D_cal ∩ D_test = ∅`

The predictive model is fitted on:
`D_train`
The probability calibrator is fitted using:
`D_cal`
and final performance is measured only on:
`D_test`

For binary prediction:
`Y ∈ {0,1}`
where:
`0 = NotLead`
`1 = Lead`

Predictive ambiguity can be represented using:
`u(p) = 1 - |2p - 1|`
while conformal prediction independently produces a prediction set:
`C(x) ⊆ {NotLead, Lead}`

The decision system then uses predictive risk, uncertainty, equity, and budget constraints to determine an acquisition policy.

The active-learning loop updates:
`L_t`
after every inspection batch and rebuilds label-dependent spatial features using the updated labeled set.

## 🏆 What Makes LeadGuard Interesting

The strongest part of LeadGuard is not:
"We used XGBoost."
XGBoost is a tool.

The interesting part is the system-level question:
**How should a city spend a finite inspection budget when the underlying infrastructure state is only partially known?**

That requires combining:
- supervised learning
- probability calibration
- uncertainty quantification
- conformal prediction
- spatial modeling
- active learning
- resource allocation
- fairness auditing
- explainability
- privacy-aware APIs
- human-in-the-loop workflows

That combination is the core of LeadGuard.

## 🧪 A Strong Evaluation Story

A credible LeadGuard evaluation should eventually look like this:

```
          REAL DATA
              │
              ▼
        Leakage Audit
              │
              ▼
      Geographic Holdout
              │
              ▼
    ┌─────────────────────┐
    │   Baseline Models   │
    └──────────┬──────────┘
               │
               ▼
         XGBoost Model
               │
               ▼
          Calibration
               │
               ▼
     Uncertainty Analysis
               │
               ▼
     Conformal Evaluation
               │
               ▼
     Decision Simulation
               │
               ▼
       Active Learning
               │
               ▼
      Budget Comparison
               │
               ▼
      Fairness Analysis
               │
               ▼
       Robustness Tests
```

A model should not be called production-ready merely because it has a high PR-AUC.
It should survive the entire pipeline.

## 🔥 The Standard LeadGuard Sets for Itself

A successful LeadGuard release should be able to answer:

- **Can we trust the evaluation?** No leakage.
- **Can we interpret the probability?** Calibration.
- **Can we identify uncertainty?** Predictive uncertainty + conformal sets.
- **Can we choose inspections intelligently?** Risk-aware acquisition.
- **Can the model learn from inspections?** Active learning.
- **Can we operate under a budget?** Budget-aware prioritization.
- **Can we monitor equity?** Decision-level fairness metrics.
- **Can humans understand the output?** Explainability.
- **Can we avoid unnecessary exposure of sensitive information?** Privacy-preserving API design.

## 📚 Technical Summary

| Component | LeadGuard Approach |
| --- | --- |
| Problem | Lead service-line inspection prioritization |
| Primary target | Binary Lead / NotLead |
| Core model | XGBoost |
| Feature type | Structured + spatial |
| Spatial features | Train/reference-only label usage |
| Evaluation | Leakage-safe holdout |
| Probability | Calibrated |
| Calibration | Held-out calibration set |
| Uncertainty | Predictive uncertainty |
| Conformal output | Binary prediction sets |
| Acquisition | Active learning |
| Baselines | Random / risk / uncertainty / combined |
| Oracle | Simulation upper bound |
| Fairness | Allocation + error/disparity analysis |
| Explainability | SHAP-compatible tree explanations |
| API | FastAPI |
| Dashboard | Streamlit |
| Privacy | No address in public priority queue |
| Testing | Unit + integration + leakage tests |
| Primary use | Inspection decision support |

## ⚠️ Responsible Use

LeadGuard should be used as a decision-support system.
Any operational deployment should include:
- human review
- appropriate inspection procedures
- domain expertise
- legal/privacy review
- data-quality monitoring
- model-performance monitoring
- calibration monitoring
- fairness monitoring
- distribution-shift monitoring

A model prediction should never be treated as physical confirmation of pipe material.

## 🤝 Contributing

Contributions are welcome.
When modifying LeadGuard, contributors should pay particular attention to:
- Data leakage
- Train/test contamination
- Calibration integrity
- Uncertainty semantics
- Privacy
- Fairness
- Reproducibility

Any new label-dependent feature should explicitly document:
*What labels does it use? From which reference population? At what stage is it calculated?*

## 📄 License

See the repository license for the applicable terms.

## 👨‍💻 Final Takeaway

LeadGuard is built around a simple idea:
**When you cannot inspect everything, the goal is not merely to predict—it is to decide what to inspect next, understand how uncertain that decision is, learn from the inspection, and do so responsibly.**

The system therefore closes the loop:

```
    ┌──────────────────────────────┐
    │                              │
    ▼                              │
 PREDICT                           │
    │                              │
    ▼                              │
CALIBRATE                          │
    │                              │
    ▼                              │
QUANTIFY UNCERTAINTY               │
    │                              │
    ▼                              │
PRIORITIZE                         │
    │                              │
    ▼                              │
 INSPECT                           │
    │                              │
    ▼                              │
  LEARN                            │
    │                              │
    └──────────────────────────────┘
```

LeadGuard is not trying to replace inspectors.
It is trying to help them answer one extremely important question:
**"Given what we know, what should we inspect next?"**

And just as importantly:
**"How sure are we—and who might we be leaving behind?"**
