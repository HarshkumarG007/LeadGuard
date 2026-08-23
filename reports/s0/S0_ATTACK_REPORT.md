# S0 Adversarial Verification Report
Protocol: LG-S1-001
S0_MACHINE_VERDICT: INCOMPLETE
HUMAN_ATTESTATION: REQUIRED
S1_AUTHORIZATION: HUMAN_SIGNATURE AND INDEPENDENT_VERIFICATION AND ZERO_CRITICAL_VIOLATIONS
Generated: 2026-08-23T20:04:31.960421+00:00
Deterministic attacks: 43 / 1300+

## Verification Metadata
- **Contract Hash**: 3080c2e617687c2f3ddc4edca30d8aad31575dc2
- **Harness Commit**: <git SHA> (simulated)
- **Harness Version**: 1.0.0
- **Attack Corpus Hash**: (Dynamically generated)
- **Manifest Hash**: 84900b8dc160e08bc4130f255bd857ec6e48b8210723f825fc9c315783f2ea11
- **Signature Identity**: RSA-2048 (Offline Key)
- **Property-based Seed Policy**: Hypothesis (default profile)
- **Total Generated Cases**: 292
- **Unique Attacks**: 43
- **Failures (Expected Rejects)**: 24
- **Critical Violations**: 0

## Coverage Matrix
| Family | Deterministic | PBE | Status |
| :--- | :--- | :--- | :--- |
| Temporal | ✅ | ✅ | ESTABLISHED |
| Feature lineage | ✅ | ⬜ | ESTABLISHED |
| Identity/replay | ✅ | ⬜ | ESTABLISHED |
| Cryptographic ledger | ✅ | ⬜ | ESTABLISHED |
| Provenance | ✅ | ⬜ | ESTABLISHED |
| Cohort snapshot | ✅ | ✅ | ESTABLISHED |
| Censoring | ✅ | ⬜ | ESTABLISHED |
| Metrics/Numerics | ✅ | ⬜ | ESTABLISHED |
| Policy/budget | ✅ | ⬜ | ESTABLISHED |
| API safety | ✅ | ⬜ | ESTABLISHED |
| Schema canonicalization | ✅ | ✅ | ESTABLISHED |
| Concurrency/clock | ✅ | ⬜ | ESTABLISHED |
*Note: PBE Coverage stands at 3/12 families.*

## Summary
The S0 Adversarial Verification Harness executed the threat families against the LeadGuard 2.0 implementation.

## Output
```text
============================= test session starts =============================
platform win32 -- Python 3.13.5, pytest-9.1.1, pluggy-1.6.0 -- C:\Users\Lenovo\Downloads\LeadGuard\.venv\Scripts\python.exe
cachedir: .pytest_cache
hypothesis profile 'default'
rootdir: C:\Users\Lenovo\Downloads\LeadGuard
configfile: pyproject.toml
plugins: anyio-4.14.2, hypothesis-6.165.10, asyncio-1.4.0, cov-7.1.0, typeguard-4.6.0
asyncio: mode=Mode.STRICT, debug=False, asyncio_default_fixture_loop_scope=None, asyncio_default_test_loop_scope=function
collecting ... collected 43 items

tests/s0/adversarial/test_api_shadow_safety.py::test_api_shadow_safety_normal PASSED [  2%]
tests/s0/adversarial/test_api_shadow_safety.py::test_api_shadow_safety_fail_closed PASSED [  4%]
tests/s0/adversarial/test_beautiful_lies.py::test_lie_4_unknown_laundering PASSED [  6%]
tests/s0/adversarial/test_beautiful_lies.py::test_lie_7_duplicate_positives PASSED [  9%]
tests/s0/adversarial/test_censoring_denominators.py::test_denominator_conservation PASSED [ 11%]
tests/s0/adversarial/test_cohort_snapshot.py::test_cohort_reorder PASSED [ 13%]
tests/s0/adversarial/test_cohort_snapshot.py::test_cohort_add PASSED     [ 16%]
tests/s0/adversarial/test_cohort_snapshot.py::test_cohort_remove PASSED  [ 18%]
tests/s0/adversarial/test_cohort_snapshot.py::test_cohort_eligibility PASSED [ 20%]
tests/s0/adversarial/test_cohort_snapshot.py::test_cohort_regulatory PASSED [ 23%]
tests/s0/adversarial/test_cohort_snapshot.py::test_cohort_duplicate PASSED [ 25%]
tests/s0/adversarial/test_cohort_snapshot.py::test_cohort_malformed PASSED [ 27%]
tests/s0/adversarial/test_concurrency_clock.py::test_clock_rollback PASSED [ 30%]
tests/s0/adversarial/test_concurrency_clock.py::test_concurrent_writers_fork PASSED [ 32%]
tests/s0/adversarial/test_concurrency_clock.py::test_double_commit PASSED [ 34%]
tests/s0/adversarial/test_concurrency_clock.py::test_crash_consistency PASSED [ 37%]
tests/s0/adversarial/test_cryptographic_ledger.py::test_ledger_drop_record PASSED [ 39%]
tests/s0/adversarial/test_cryptographic_ledger.py::test_ledger_recomputed_fake_chain PASSED [ 41%]
tests/s0/adversarial/test_feature_lineage.py::test_feature_lineage_leak PASSED [ 44%]
tests/s0/adversarial/test_feature_lineage.py::test_feature_lineage_valid PASSED [ 46%]
tests/s0/adversarial/test_identity_replay.py::test_identity_duplicate_valid_observations PASSED [ 48%]
tests/s0/adversarial/test_identity_replay.py::test_identity_replay_old_decision PASSED [ 51%]
tests/s0/adversarial/test_mutation_effectiveness.py::test_mutation_effectiveness_temporal_shift PASSED [ 53%]
tests/s0/adversarial/test_mutation_effectiveness.py::test_mutation_effectiveness_identity_swap PASSED [ 55%]
tests/s0/adversarial/test_pbe_expansion.py::test_pbe_temporal_invariant PASSED [ 58%]
tests/s0/adversarial/test_pbe_expansion.py::test_metamorphic_input_permutation PASSED [ 60%]
tests/s0/adversarial/test_pbe_expansion.py::test_metamorphic_ineligible_addition PASSED [ 62%]
tests/s0/adversarial/test_pbe_expansion.py::test_pbe_canonicalization PASSED [ 65%]
tests/s0/adversarial/test_policy_budget.py::test_policy_budget_invariance PASSED [ 67%]
tests/s0/adversarial/test_policy_budget.py::test_policy_deterministic_tiebreaker PASSED [ 69%]
tests/s0/adversarial/test_policy_budget.py::test_counterfactual_budget_invariance_random_seed PASSED [ 72%]
tests/s0/adversarial/test_provenance.py::test_provenance_model_hash_mismatch PASSED [ 74%]
tests/s0/adversarial/test_provenance.py::test_provenance_missing_hash PASSED [ 76%]
tests/s0/adversarial/test_schema_canonicalization.py::test_canonicalize_json_key_order PASSED [ 79%]
tests/s0/adversarial/test_schema_canonicalization.py::test_canonicalize_json_spacing PASSED [ 81%]
tests/s0/adversarial/test_schema_canonicalization.py::test_canonicalize_json_utf8 PASSED [ 83%]
tests/s0/adversarial/test_schema_canonicalization.py::test_canonicalize_json_nested PASSED [ 86%]
tests/s0/adversarial/test_schema_canonicalization.py::test_canonicalize_null_vs_absent PASSED [ 88%]
tests/s0/adversarial/test_schema_canonicalization.py::test_canonicalize_negative_zero PASSED [ 90%]
tests/s0/adversarial/test_schema_canonicalization.py::test_canonicalize_nan_infinity_rejected PASSED [ 93%]
tests/s0/adversarial/test_temporal.py::test_temporal_feature_leak PASSED [ 95%]
tests/s0/adversarial/test_temporal.py::test_temporal_decision_overlap PASSED [ 97%]
tests/s0/adversarial/test_temporal.py::test_temporal_timezone_naive PASSED [100%]

============================= 43 passed in 0.92s ==============================

```
