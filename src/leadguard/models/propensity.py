import logging
import argparse
import pandas as pd
import numpy as np
from pathlib import Path
import xgboost as xgb
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score, average_precision_score

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(name)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)

# Intrinsic and Geographic features (Groups A and B)
PROPENSITY_FEATURES = [
    "year_built",
    "lot_size_sqft",
    "building_sqft",
    "stories",
    "has_basement",
    "dist_to_nearest_hydrant_m",
]

def train_propensity_model(features_path: Path | str, output_dir: Path | str):
    features_path = Path(features_path)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    logger.info("Loading data from %s", features_path)
    df = pd.read_parquet(features_path)
    
    # Target: 1 if inspected, 0 if not
    # In the dataset, inspected_at is present when the property was inspected
    from leadguard.models.serving import KNOWN_MATERIALS
    df["is_inspected"] = df["service_line_material"].isin(KNOWN_MATERIALS).astype(int)
    
    logger.info("Total rows: %d, Inspected: %d", len(df), df["is_inspected"].sum())
    
    # We need to fill NaNs in features
    # For propensity, we'll just use simple imputation or let XGBoost handle it
    from leadguard.data.validation import validate_features
    X = validate_features(df, PROPENSITY_FEATURES).values
    y = df["is_inspected"].values
    
    # Split
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
    
    # Class weight
    scale_pos_weight = float((y_train == 0).sum() / (y_train == 1).sum()) if (y_train == 1).sum() > 0 else 1.0
    
    logger.info("Training XGBoost Propensity Model (scale_pos_weight=%.2f)", scale_pos_weight)
    
    model = xgb.XGBClassifier(
        n_estimators=100,
        max_depth=5,
        learning_rate=0.1,
        scale_pos_weight=scale_pos_weight,
        random_state=42,
        eval_metric="aucpr",
    )
    
    model.fit(X_train, y_train)
    
    # Predict
    test_proba = model.predict_proba(X_test)[:, 1]
    
    roc_auc = roc_auc_score(y_test, test_proba)
    pr_auc = average_precision_score(y_test, test_proba)
    baseline_pr = y_test.mean()
    
    logger.info("Evaluation on TEST set:")
    logger.info("ROC-AUC: %.4f", roc_auc)
    logger.info("PR-AUC:  %.4f (baseline = %.4f)", pr_auc, baseline_pr)
    
    # Save Model
    model_path = output_dir / "propensity_model.json"
    model.save_model(model_path)
    logger.info("Model saved to %s", model_path)
    
    return {"roc_auc": roc_auc, "pr_auc": pr_auc}

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--features", default="data/processed/features_sample.parquet")
    parser.add_argument("--output-dir", default="artifacts/c3_propensity")
    args = parser.parse_args()
    train_propensity_model(args.features, args.output_dir)
