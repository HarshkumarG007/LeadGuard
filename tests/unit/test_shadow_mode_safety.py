"""Tests for the Shadow Mode / Production Mode capability boundary (S0.4)."""

import os
import sys
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../../')))
from unittest import mock
import pytest
from fastapi.testclient import TestClient
from api.main import app

@pytest.fixture
def client():
    return TestClient(app)

def test_production_mode_requires_explicit_enablement(client):
    """Test that absence of production capability is the default."""
    # Ensure neither env var is set
    with mock.patch.dict(os.environ, {}, clear=True):
        response = client.post(
            "/v1/decisions/issue",
            json={
                "property_ids": ["prop1", "prop2"],
                "decision_type": "inspection",
                "reasoning": "Test reasoning"
            },
            headers={"Idempotency-Key": "test-key-1"}
        )
        assert response.status_code == 403
        data = response.json()
        assert "Operational dispatch disabled" in data["detail"]["error"]
        assert "LEADGUARD_PRODUCTION_MODE is not explicitly enabled" in data["detail"]["detail"]

def test_shadow_mode_cannot_dispatch(client):
    """Test that SHADOW_MODE explicitly prevents operational dispatch."""
    with mock.patch.dict(os.environ, {"LEADGUARD_SHADOW_MODE": "true"}, clear=True):
        response = client.post(
            "/v1/decisions/issue",
            json={
                "property_ids": ["prop1", "prop2"],
                "decision_type": "inspection",
                "reasoning": "Test reasoning"
            },
            headers={"Idempotency-Key": "test-key-2"}
        )
        assert response.status_code == 403
        data = response.json()
        assert "Operational dispatch disabled" in data["detail"]["error"]

def test_production_mode_can_dispatch(client):
    """Test that explicit enablement allows dispatch."""
    with mock.patch.dict(os.environ, {"LEADGUARD_PRODUCTION_MODE": "true"}, clear=True):
        # We also need to mock ImmutableLedger to avoid writing to real files during test
        with mock.patch("leadguard.data.ledger.ImmutableLedger.append_event") as mock_append:
            response = client.post(
                "/v1/decisions/issue",
                json={
                    "property_ids": ["prop1", "prop2"],
                    "decision_type": "inspection",
                    "reasoning": "Test reasoning"
                },
                headers={"Idempotency-Key": "test-key-3"}
            )
            assert response.status_code == 200
            data = response.json()
            assert data["status"] == "success"
            assert "dec-" in data["decision_id"]
            mock_append.assert_called_once()
