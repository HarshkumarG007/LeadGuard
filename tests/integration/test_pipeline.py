"""Integration tests for the full pipeline.

These tests ensure the full pipeline runs successfully on sample data and provide coverage
for the module entry points.
"""

from pathlib import Path
from unittest.mock import patch


def test_full_pipeline_coverage(tmp_path):
    """Run all phases through their module entry points to guarantee integration works."""
    import os

    from leadguard.data.clean import main as clean_main
    from leadguard.data.features import main as features_main
    from leadguard.evaluation.explainability import main as explainability_main
    from leadguard.evaluation.fairness import main as fairness_main
    from leadguard.models.active_learning import main as active_learning_main
    from leadguard.models.baseline import main as baseline_main
    from leadguard.models.uncertainty import main as uncertainty_main
    from leadguard.models.xgboost_model import main as xgboost_main
    os.environ["DISABLE_PANDERA_IMPORT_WARNING"] = "True"

    interim = str(tmp_path / "interim.parquet")
    features = str(tmp_path / "features.parquet")
    model_out = tmp_path / "models" / "xgboost"

    # Mock OSM hydrants to prevent FileNotFoundError
    with patch("leadguard.data.features._load_osm_hydrants") as mock_hydrants:
        import pandas as pd
        mock_hydrants.return_value = pd.DataFrame({"latitude": [41.8], "longitude": [-87.6]})

        # Create dummy census data for fairness script
        raw_dir = tmp_path / "raw"
        raw_dir.mkdir()
        (raw_dir / "census_acs_cook_county.csv").write_text('census_tract,median_household_income\n17031010100,50000')

        # 1. Clean
        with patch("sys.argv", ["clean.py", "--input", "data/sample", "--output", interim]):
            clean_main()
        assert Path(interim).exists(), "Clean stage failed to produce interim output"

        # 2. Features
        with patch("sys.argv", ["features.py", "--input", interim, "--output", features]):
            features_main()
        assert Path(features).exists(), "Features stage failed to produce output"

        # 3. Baseline
        with patch(
            "sys.argv", ["baseline.py", "--config", "configs/train.yaml", "--features", features]
        ):
            baseline_main()
        assert Path("reports/baseline_metrics.json").exists(), "Baseline failed"

        # 4. XGBoost
        with patch(
            "sys.argv", ["xgboost_model.py", "--config", "configs/train.yaml", "--features", features, "--output-dir", str(model_out)]
        ):
            import yaml

            with open("configs/train.yaml") as f:
                cfg = yaml.safe_load(f)
            cfg["optuna"] = {"n_trials": 1, "timeout_seconds": 60}

            with patch("leadguard.models.xgboost_model.yaml.safe_load") as mock_yaml:
                mock_yaml.return_value = cfg
                xgboost_main()

        assert (model_out / "xgb_model.pkl").exists(), "XGBoost failed to save calibrated artifact"
        assert (model_out / "model.json").exists(), "XGBoost failed to save raw model"
        assert (model_out / "metadata.json").exists(), "XGBoost failed to save metadata"

        # 5. Uncertainty
        with patch("sys.argv", ["uncertainty.py", "--features", features, "--output-dir", str(model_out)]):
            uncertainty_main()
        assert (model_out / "conformal_global.pkl").exists(), "Uncertainty failed to save conformal predictor"

        # 6. Fairness
        with patch("sys.argv", ["fairness.py", "--sample", "--features", features, "--raw-dir", str(raw_dir), "--model-dir", str(model_out)]):
            fairness_main()

        # 7. Active Learning
        with patch("sys.argv", ["active_learning.py", "--fast", "--features", features]):
            active_learning_main()

        # 8. Explainability
        with patch("sys.argv", ["explainability.py", "--features", features, "--model-dir", str(model_out)]):
            explainability_main()
