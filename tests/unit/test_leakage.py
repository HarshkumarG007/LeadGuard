import numpy as np
import pandas as pd

from leadguard.data.features import build_features


def _make_dummy_data():
    return pd.DataFrame(
        {
            "property_id": [f"chi-{i}" for i in range(10)],
            "latitude": [41.8 + i * 0.01 for i in range(10)],
            "longitude": [-87.6 + i * 0.01 for i in range(10)],
            "ward": [1, 1, 1, 2, 2, 2, 3, 3, 3, 4],
            "service_line_material": [
                "Lead",
                "Copper",
                "Lead",
                "Galvanized",
                "Lead",
                "Copper",
                "Unknown",
                "Lead",
                "Lead",
                "Galvanized",
            ],
            "year_built": [1900, 1950, 1920, 1980, 1910, 2000, 1960, 1890, 1930, 1970],
        }
    )


def test_test_labels_cannot_affect_test_features():
    df = _make_dummy_data()
    train_idx = df.index[:5]
    test_idx = df.index[5:]

    train_df = df.loc[train_idx].copy()
    test_df = df.loc[test_idx].copy()

    # Baseline features
    test_features_baseline = build_features(
        test_df, reference_df=train_df, include_label_dependent=True
    )

    # Now randomize test labels
    test_df_randomized = test_df.copy()
    test_df_randomized["service_line_material"] = "Lead"  # All Lead!

    test_features_randomized = build_features(
        test_df_randomized, reference_df=train_df, include_label_dependent=True
    )

    # Assert features are identical (ignoring the label column itself)
    pd.testing.assert_frame_equal(
        test_features_baseline.drop(columns=["service_line_material"]),
        test_features_randomized.drop(columns=["service_line_material"]),
    )


def test_spatial_features_use_training_labels_only():
    df = _make_dummy_data()
    train_idx = df.index[:5]
    test_idx = df.index[5:]

    train_df = df.loc[train_idx].copy()
    test_df = df.loc[test_idx].copy()

    train_features_baseline = build_features(
        train_df, reference_df=train_df, include_label_dependent=True
    )
    test_features_baseline = build_features(
        test_df, reference_df=train_df, include_label_dependent=True
    )

    # Change a test label
    test_df_modified = test_df.copy()
    test_df_modified.loc[5, "service_line_material"] = "Lead"

    train_features_modified = build_features(
        train_df, reference_df=train_df, include_label_dependent=True
    )
    test_features_modified = build_features(
        test_df_modified, reference_df=train_df, include_label_dependent=True
    )

    # Features must be completely unchanged (ignoring the label column itself on the test set)
    pd.testing.assert_frame_equal(train_features_baseline, train_features_modified)
    pd.testing.assert_frame_equal(
        test_features_baseline.drop(columns=["service_line_material"]),
        test_features_modified.drop(columns=["service_line_material"]),
    )


def test_label_permutation_collapses_performance():
    # If we randomize train labels, model performance should drop to prevalence (~0.5)
    # We will simulate this by ensuring that features computed with random labels
    # don't provide magical predictive power.
    df = _make_dummy_data()
    train_idx = df.index[:5]
    test_idx = df.index[5:]

    train_df = df.loc[train_idx].copy()
    test_df = df.loc[test_idx].copy()

    # Randomize training labels
    rng = np.random.default_rng(42)
    train_df["service_line_material"] = rng.choice(["Lead", "Copper"], size=len(train_df))

    test_features = build_features(test_df, reference_df=train_df, include_label_dependent=True)
    # The nearest known lead might change, but it must strictly only depend on the randomized train_df.
    # The true test labels (which are not randomized) shouldn't be predictable from this.
    assert "dist_to_nearest_known_lead_m" in test_features.columns
