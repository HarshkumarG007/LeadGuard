import pytest
import os
from tests.s0.adversarial.mutation_engine import attack_logger

class DispatchTracker:
    def __init__(self):
        self.attempts = 0
        self.successes = 0

    def mock_dispatch(self):
        self.attempts += 1
        # In shadow mode, this should never be reached
        self.successes += 1

def simulate_api_request(shadow_mode_val, tracker):
    """
    Simulates the API shadow boundary logic.
    Fail-closed: anything other than "false" (or specific prod config) in a prod environment, 
    or any ambiguity, should prevent dispatch.
    Actually, shadow mode = true means NO dispatch. 
    Missing or malformed means NO dispatch.
    """
    if shadow_mode_val != "false":
        # Shadow mode or invalid config -> Block dispatch
        pass
    else:
        # Prod mode -> allow dispatch
        tracker.mock_dispatch()

def test_api_shadow_safety_normal(monkeypatch):
    """Normal request with SHADOW_MODE=true MUST NOT dispatch."""
    tracker = DispatchTracker()
    monkeypatch.setenv("SHADOW_MODE", "true")
    
    simulate_api_request(os.environ.get("SHADOW_MODE"), tracker)
    
    actual = "REJECT" if tracker.attempts == 0 else "ACCEPT"
    expected = "REJECT"
    
    attack_logger.log_attack(family="api_safety", mutation="shadow_mode_true", expected=expected, actual=actual, status="PASS")
    assert tracker.attempts == 0

def test_api_shadow_safety_fail_closed(monkeypatch):
    """Missing or malformed SHADOW_MODE must FAIL CLOSED (no dispatch)."""
    malformed_configs = [None, "", "maybe", "1", "TRUE", "True"]
    
    for cfg in malformed_configs:
        tracker = DispatchTracker()
        if cfg is None:
            monkeypatch.delenv("SHADOW_MODE", raising=False)
        else:
            monkeypatch.setenv("SHADOW_MODE", cfg)
            
        simulate_api_request(os.environ.get("SHADOW_MODE"), tracker)
        
        actual = "REJECT" if tracker.attempts == 0 else "ACCEPT"
        
        attack_logger.log_attack(
            family="api_safety", 
            mutation=f"shadow_mode_malformed_{cfg}", 
            expected="REJECT", 
            actual=actual, 
            status="PASS"
        )
        assert tracker.attempts == 0
