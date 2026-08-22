import pandas as pd
import pytest

from leadguard.data.validation import FeatureContractError, validate_features


def test_validate_features_success():
    df = pd.DataFrame({"feat_a": [1, 2], "feat_b": [3, 4], "extra": [5, 6]})
    validated = validate_features(df, ["feat_a", "feat_b"])
    assert list(validated.columns) == ["feat_a", "feat_b"]
    assert len(validated) == 2

def test_validate_features_missing_raises_error():
    df = pd.DataFrame({"feat_a": [1, 2]})
    with pytest.raises(FeatureContractError) as exc:
        validate_features(df, ["feat_a", "feat_missing"])

    assert "feat_missing" in str(exc.value)
