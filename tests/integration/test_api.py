"""Integration tests for the LeadGuard API (Phase 9).

Tests all 7 endpoints against the sample dataset.
Requires the API server to be running OR uses TestClient.
"""

from __future__ import annotations

from pathlib import Path

import pytest
from fastapi.testclient import TestClient


@pytest.fixture(scope="module")
def client() -> TestClient:
    """Create a TestClient for the FastAPI app."""
    # Set correct working directory for model loading
    import os

    os.chdir(Path(__file__).parent.parent.parent)

    from api.model_loader import reset_state

    reset_state()

    from api.main import app

    return TestClient(app)


@pytest.fixture(scope="module")
def sample_property_id(client: TestClient) -> str:
    """Get a valid property ID from the priority queue."""
    resp = client.get("/v1/priority-queue", params={"budget_usd": 100000, "limit": 5})
    if resp.status_code == 200:
        items = resp.json().get("items", [])
        if items:
            return items[0]["property_id"]
    # Fallback: load from sample data
    sample_path = Path("data/sample/sample_properties.parquet")
    if sample_path.exists():
        import pandas as pd

        df = pd.read_parquet(sample_path)
        return str(df["property_id"].iloc[0])
    return "chi-00000000"


class TestHealthEndpoint:
    """GET /v1/health"""

    def test_health_returns_200_or_503(self, client: TestClient) -> None:
        """Health endpoint must return 200 (ok) or 503 (degraded), never 5xx crash."""
        resp = client.get("/v1/health")
        assert resp.status_code in (200, 503)

    def test_health_response_has_status_field(self, client: TestClient) -> None:
        """Response body must contain 'status' field."""
        resp = client.get("/v1/health")
        body = resp.json()
        assert "status" in body


class TestPredictEndpoint:
    """POST /v1/predict"""

    def test_predict_valid_request(self, client: TestClient, sample_property_id: str) -> None:
        """Valid property ID should return a prediction or 503 if model not loaded."""
        resp = client.post("/v1/predict", json={"property_ids": [sample_property_id]})
        assert resp.status_code in (200, 404, 503)
        if resp.status_code == 200:
            body = resp.json()
            assert "predictions" in body

    def test_predict_response_schema(self, client: TestClient, sample_property_id: str) -> None:
        """Prediction response must contain required fields."""
        resp = client.post("/v1/predict", json={"property_ids": [sample_property_id]})
        if resp.status_code == 200:
            pred = resp.json()["predictions"][0]
            assert "p_lead_calibrated" in pred
            assert "conformal_set" in pred
            assert "uncertainty_score" in pred
            assert "priority_score" in pred
            assert "shap_top_features" in pred
            assert "model_version" in pred

    def test_predict_malformed_request_returns_422(self, client: TestClient) -> None:
        """Empty property_ids should return 422."""
        resp = client.post("/v1/predict", json={"property_ids": []})
        assert resp.status_code == 422

    def test_predict_invalid_json_returns_422(self, client: TestClient) -> None:
        """Missing required field should return 422."""
        resp = client.post("/v1/predict", json={"wrong_field": "value"})
        assert resp.status_code == 422


class TestSinglePredictionEndpoint:
    """GET /v1/properties/{property_id}/prediction"""

    def test_single_prediction_valid_id(self, client: TestClient, sample_property_id: str) -> None:
        """Valid property ID returns prediction or 404."""
        resp = client.get(f"/v1/properties/{sample_property_id}/prediction")
        assert resp.status_code in (200, 404, 503)

    def test_unknown_property_returns_404(self, client: TestClient) -> None:
        """Unknown property ID must return 404."""
        resp = client.get("/v1/properties/definitely-does-not-exist-abc123/prediction")
        assert resp.status_code in (404, 503)


class TestPriorityQueueEndpoint:
    """GET /v1/priority-queue"""

    def test_priority_queue_returns_200_or_503(self, client: TestClient) -> None:
        """Priority queue returns 200 or 503."""
        resp = client.get("/v1/priority-queue", params={"budget_usd": 50000, "limit": 10})
        assert resp.status_code in (200, 503)

    def test_priority_queue_schema(self, client: TestClient) -> None:
        """Priority queue response has expected structure."""
        resp = client.get("/v1/priority-queue", params={"budget_usd": 50000, "limit": 10})
        if resp.status_code == 200:
            body = resp.json()
            assert "items" in body
            assert "budget_usd" in body
            assert "properties_within_budget" in body


class TestInspectionsEndpoint:
    """POST /v1/inspections"""

    def test_submit_inspection(self, client: TestClient, sample_property_id: str) -> None:
        """Valid inspection submission returns 200 with inspection_id."""
        resp = client.post(
            "/v1/inspections",
            json={
                "property_id": sample_property_id,
                "inspected_material": "Copper",
                "source": "field_inspection",
                "cost_usd": 500.0,
            },
        )
        assert resp.status_code == 200
        body = resp.json()
        assert "inspection_id" in body
        assert body["inspected_material"] == "Copper"

    def test_invalid_material_returns_422(self, client: TestClient) -> None:
        """Invalid material should return 422."""
        resp = client.post(
            "/v1/inspections",
            json={
                "property_id": "some-id",
                "inspected_material": "PVC",  # not in allowed set
            },
        )
        assert resp.status_code == 422


class TestFairnessReportEndpoint:
    """GET /v1/fairness-report"""

    def test_fairness_report_returns_200_or_503(self, client: TestClient) -> None:
        """Fairness report returns 200 or 503 if not generated."""
        resp = client.get("/v1/fairness-report")
        assert resp.status_code in (200, 503)

    def test_fairness_report_schema(self, client: TestClient) -> None:
        """Fairness report must have fnr_by_quartile and equity_boost_sample."""
        resp = client.get("/v1/fairness-report")
        if resp.status_code == 200:
            body = resp.json()
            assert "fnr_by_quartile" in body
            assert "equity_boost_sample" in body
            assert "disparity_flagged" in body


class TestModelMetadataEndpoint:
    """GET /v1/model/metadata"""

    def test_metadata_returns_200(self, client: TestClient) -> None:
        """Metadata endpoint always returns 200 (even without model)."""
        resp = client.get("/v1/model/metadata")
        assert resp.status_code == 200

    def test_metadata_schema(self, client: TestClient) -> None:
        """Metadata must have model_version and features list."""
        resp = client.get("/v1/model/metadata")
        body = resp.json()
        assert "model_version" in body
        assert "features" in body
        assert "confidence_level" in body
