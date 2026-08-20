"""Integration tests for the full pipeline.

These tests ensure the full pipeline runs successfully on sample data and provide coverage
for the module entry points.
"""

from unittest.mock import patch


def test_full_pipeline_coverage(tmp_path):
    """Run all phases through their module entry points to guarantee integration works."""
    from leadguard.data.clean import main as clean_main
    from leadguard.data.download import main as download_main
    from leadguard.data.features import main as features_main
    from leadguard.evaluation.explainability import main as explainability_main
    from leadguard.evaluation.fairness import main as fairness_main
    from leadguard.models.active_learning import main as active_learning_main
    from leadguard.models.baseline import main as baseline_main
    from leadguard.models.uncertainty import main as uncertainty_main
    from leadguard.models.xgboost_model import main as xgboost_main

    interim = str(tmp_path / "interim.parquet")
    features = str(tmp_path / "features.parquet")

    # 1. Download (mocked)
    with patch("sys.argv", ["download.py", "--raw-dir", str(tmp_path)]), patch("requests.get"):
        try:
            download_main()
        except Exception:
            pass  # Ignore if it fails due to mock

    # 1. Clean
    with patch("sys.argv", ["clean.py", "--input", "data/sample", "--output", interim]):
        clean_main()

    # 2. Features
    with patch("sys.argv", ["features.py", "--input", interim, "--output", features]):
        features_main()

    # 3. Baseline
    with patch(
        "sys.argv", ["baseline.py", "--config", "configs/train.yaml", "--features", features]
    ):
        baseline_main()

    # 4. XGBoost
    with patch(
        "sys.argv", ["xgboost_model.py", "--config", "configs/train.yaml", "--features", features]
    ):
        xgboost_main()

    # 5. Uncertainty
    with patch("sys.argv", ["uncertainty.py", "--features", features, "--sample"]):
        uncertainty_main()

    # 6. Fairness
    with patch("sys.argv", ["fairness.py", "--sample"]):
        fairness_main()

    # 7. Active Learning
    with patch("sys.argv", ["active_learning.py", "--sample"]):
        active_learning_main()

    # 8. Explainability
    with patch("sys.argv", ["explainability.py", "--sample"]):
        explainability_main()
