import time
import pytest
from fastapi.testclient import TestClient
from api.main import app, _get_properties, get_state
from leadguard.data.validation import validate_features

client = TestClient(app)

@pytest.fixture(autouse=True)
def setup_api():
    """Ensure the state is loaded before tests."""
    get_state()
    _get_properties()

def test_api_latency_bounded():
    """Test that online prediction is fast and bounded because of precomputation."""
    props = _get_properties()
    if props.empty:
        pytest.skip("No properties data available for performance test.")
        
    test_ids = props["property_id"].head(50).tolist()
    
    latencies = []
    
    # Warm up cache
    for pid in test_ids[:5]:
        client.get(f"/v1/properties/{pid}/prediction")
        
    # Benchmark
    for pid in test_ids:
        start_time = time.perf_counter()
        response = client.get(f"/v1/properties/{pid}/prediction")
        end_time = time.perf_counter()
        
        assert response.status_code == 200
        latencies.append((end_time - start_time) * 1000) # ms
        
    latencies.sort()
    p50 = latencies[int(len(latencies) * 0.50)]
    p95 = latencies[int(len(latencies) * 0.95)]
    p99 = latencies[int(len(latencies) * 0.99)]
    
    print(f"\nWarm-cache benchmark (n={len(test_ids)}):")
    print(f"p50: {p50:.2f} ms")
    print(f"p95: {p95:.2f} ms")
    print(f"p99: {p99:.2f} ms")
    
    # Bound assertion - if SHAP is precomputed or O(1), this should be extremely fast
    assert p95 < 500, f"Latency unbounded! p95 was {p95:.2f} ms"
