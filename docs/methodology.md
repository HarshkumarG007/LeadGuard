# LeadGuard — Modeling Methodology

## 1. Problem Formulation
Predicting lead service line material is formulated as a binary classification problem (`Lead` vs `Not Lead`), where `Copper` and `Galvanized` are treated as the negative class. However, because field inspections are expensive and the goal is to sequence them optimally, a point probability is insufficient. We need a holistic `priority_score` that balances:
1. **Risk:** The probability of finding lead.
2. **Uncertainty:** The model's epistemic uncertainty about a prediction.
3. **Equity:** Ensuring disadvantaged or historically under-inspected neighborhoods are not systematically deprioritized.

## 2. Model Selection
We use **XGBoost** with a histogram tree method (`tree_method="hist"`). 
- **Why not Deep Learning?** Tabular datasets with categorical and spatial features typically favor tree ensembles. TabNet was considered, but XGBoost delivers state-of-the-art performance on this type of data while maintaining CPU-friendly training and rapid inference (<1ms).
- **Monotonic Constraints:** We enforce a monotonic constraint on `year_built` (-1), ensuring the model strictly associates older homes with higher lead probability, preventing overfitting to noise in the training set.

## 3. Uncertainty Quantification
We use **Conformal Prediction** to generate prediction sets with guaranteed marginal coverage (e.g., 90%). 
- **Global Split Conformal:** Calibrates a global threshold on a held-out set.
- **Mondrian Conformal:** Calibrates thresholds separately per income quartile (derived from Census ACS) to ensure that the 90% coverage guarantee holds within each socioeconomic group, not just on average.

The `uncertainty_score` is derived from the size of the conformal prediction set. A set containing all classes yields `1.0`, while a set resolving to a single class yields `0.0`.

## 4. Equity and Fairness
Protected-class-correlated fields (e.g., median household income, race) are **strictly forbidden** from the model's feature matrix to prevent direct bias encoding. 
Instead, we apply an `equity_boost` during priority queue generation. This term boosts the priority of properties located in census tracts that have received fewer field inspections relative to their model-estimated risk share.

## 5. Active Learning
The system simulates an active learning loop, prioritizing inspections where the model is most uncertain. By feeding these new ground-truth labels back into training, the model efficiently explores the feature space, leading to faster PR-AUC convergence compared to random sampling.

## 6. Explainability
Every prediction served by the API includes a SHAP (SHapley Additive exPlanations) breakdown. We use `TreeExplainer` for rapid (<100ms) extraction of the top 5 contributing features for any given property, providing transparency to city planners and residents.
