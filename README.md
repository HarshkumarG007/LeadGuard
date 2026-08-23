# 🛡️ LeadGuard 2.0

**LeadGuard 1.0 solved evaluation integrity. LeadGuard 2.0 is a cryptographically verifiable scientific apparatus for measuring utility.**

```text
┌──────────────────────────────────────────────────────────┐
│                 LEADGUARD 2.0 STATUS                     │
├──────────────────────────────────────────────────────────┤
│ S0 Verification:       COMPLETE                          │
│ S0 Artifacts:          FROZEN                            │
│ Critical Violations:   0                                 │
│ S1 Preflight:          PASS                              │
│ Independent Audit:     PENDING                           │
│ S1 Authorization:      LOCKED                            │
│ Real-World Data:       NOT ADMITTED                      │
└──────────────────────────────────────────────────────────┘
```

LeadGuard is an end-to-end machine-learning system for helping municipalities prioritize inspections for drinking-water service lines of unknown material. 

While LeadGuard 1.0 focused on eliminating spatial label leakage and building a defensible baseline (PR-AUC ≈ 0.41), **LeadGuard 2.0 structurally enforces the separation between a model's statistical beliefs and a system's real-world economic utility.**

---

## 1. The Verification Architecture

The architecture relies on a fundamental separation of powers: the machinery that makes decisions cannot be trusted to independently authorize or validate itself. The pipeline is locked until human attestation unlocks it.

```text
             ┌─────────────────────────┐
             │   S1 COHORT CONTRACT    │ (Frozen Mathematical Definitions)
             └───────────┬─────────────┘
                         │
             ┌───────────▼─────────────┐
             │   ADVERSARIAL HARNESS   │ (S0 Machine Verification)
             └───────────┬─────────────┘
                         │
             ┌───────────▼─────────────┐
             │    IMMUTABLE EVIDENCE   │ (Manifests, Ledger, Hashes)
             └───────────┬─────────────┘
                         │
             ┌───────────▼─────────────┐
             │ INDEPENDENT VERIFIER    │ (Auditor Environment & Offline Key)
             └───────────┬─────────────┘
                         │
             ┌───────────▼─────────────┐
             │    HUMAN ATTESTATION    │ (Ceremony & S1_ATTESTATION_RECORD)
             └───────────┬─────────────┘
                         │
             ┌───────────▼─────────────┐
             │      S1 LOCKED GATE     │ (ZERO-DATA PREFLIGHT)
             └───────────┬─────────────┘
                         │
                         ▼
                REAL-WORLD COHORT S1
```

---

## 2. What S0 Does and Does Not Establish

We have successfully completed Phase S0, constructing the observational infrastructure required for Phase S1. However, this is a **scientific provenance document**, not a victory report.

### **S0 establishes:**
* **Invariant enforcement**: Relational logic behaves correctly even under hostile permutation.
* **Adversarial test machinery**: A deterministic matrix covering 12 threat families.
* **Evidence integrity**: Hashes, immutability, and protection against Hostile-Maintainer attacks.
* **Shadow-mode safety boundaries**: Production endpoints demonstrably cannot dispatch.
* **Preflight readiness**: The synthetic zero-data test successfully proved isolated execution.

### **S0 does not establish:**
* **Real-world predictive performance**: Synthetic tests do not establish generalization.
* **Causal superiority**: Simulation does not equal true policy advantage.
* **Realized economic value**: Mathematical sensitivity analysis does not guarantee municipal value.
* **Equity performance on municipal data**: Real-world demographic impact remains untested.
* **Authorization for real interventions**: S1 generates shadow decisions only.

---

## 3. The Mathematical Estimands (S1)

The entire apparatus is built to correctly record data so that we can ultimately calculate the S1 estimands. These are the equations that will govern LeadGuard's utility, provided the infrastructure is proven secure.

**1. The Strict Temporal Barrier**  
Outcomes must never leak into the decision space:
$$ T_{feature} \le T_{decision} < T_{observation} \le T_{availability} $$

**2. Economic Value of Information (EVI)**  
The core hypothesis of LeadGuard 2.0 is measuring whether the predicted value translates to reality:
$$ EVI_{predicted} \longrightarrow EVI_{realized} $$

**3. Policy Regret**  
The loss function for the operational system compares the policy's chosen intervention against an omniscient oracle:
$$ Regret_{policy} = Utility_{oracle} - Utility_{policy} $$

---

## 4. The S0 Adversarial Harness

To trust the S1 measurements, the machinery recording them was subjected to an adversarial gauntlet (`scripts/run_s0_adversarial.py`).

1. **12 Threat Families**: The harness attacks temporal leakage, budget exhaustion, ledger mutability, API shadow failures, and concurrency/clock chaos.
2. **Property-Based Expansion (PBE)**: Using Hypothesis, metamorphic testing forces relational invariants to prove consistency (e.g., $f(perm(X)) == f(X)$). **Current explicit PBE coverage is 3/12 families.**
3. **Zero Critical Violations**: The S0 harness executes deterministically with 0 false positives and 0 missed breaches.

---

## 5. Air-Gapped Trust & Cryptography

S0 stripped signing authority from the operational repository. The `sign_s0_report.py` tool requires an offline RSA-2048 private key, while the independent verifier (`verify_s0_report.py`) requires only the public key.

**Hostile-Maintainer Protection**: If a compromised engineer alters the test logic to force a `PASS` and recalculates the report hash, the independent verifier immediately detects the forgery, because the private key is missing from the environment. Cryptography here guarantees artifact integrity, while **independence comes purely from the human governance procedure**.

---

## 6. Economic Sensitivity (EVI)

A sensitivity analysis (`scripts/evi_sensitivity_analysis.py`) traced the boundaries of the assumed economics. For an assumed $100 inspection cost and $5,000 intervention value, the **break-even point is $500**. 

Under the registered assumptions, this yields a **400% normalized margin of robustness**. This is an analytical boundary under predefined parameters, not empirical evidence of real-world value. See the [Assumption Register](contracts/S1_ASSUMPTION_REGISTER.md).

---

## 7. The S1 Zero-Data Preflight

Before accepting real-world data, the pipeline executed a zero-data check (`scripts/s1_zero_data_check.py`) using a synthetic namespace. 

This test established:
1. Zero outbound production dispatches.
2. The real S1 production ledger remained mathematically unmodified (hash-identical).
3. The kill-switch successfully aborted state pre-commit and recorded an auditable terminal state post-persistence.
4. Idempotency correctly handled duplicate decision attempts across processes.

---

## 8. Artifact & Git Tree

```text
LeadGuard/
├── contracts/
│   ├── s1_cohort_contract.md         # Frozen mathematical estimands
│   ├── S1_ASSUMPTION_REGISTER.md     # Frozen economic parameters
│   └── S1_ATTESTATION_RECORD.md      # Pending Human Signature
├── keys/
│   ├── human_offline_key.pub         # Public verification material
│   └── human_offline_key.pem         # NEVER IN REPOSITORY (Air-gapped)
├── reports/
│   ├── s0/
│   │   ├── S0_ATTACK_MANIFEST.jsonl  # The complete adversarial matrix
│   │   └── S0_ATTACK_REPORT.md       # Evidence report (.sha256, .sig)
│   └── s1/
│       └── S1_PREFLIGHT_REPORT.md    # Zero-data check results
├── scripts/
│   ├── evi_sensitivity_analysis.py
│   ├── hostile_maintainer_test.py
│   ├── run_s0_adversarial.py
│   ├── s1_zero_data_check.py
│   ├── sign_s0_report.py
│   └── verify_s0_report.py
└── src/leadguard/                    # The operational codebase
```

---

## 9. Next Steps

LeadGuard's operational state is currently **LOCKED**. The system cannot ingest real-world data or execute shadow decisions until the independent auditor performs the attestation ceremony, independently verifies the hashes, signs the `S1_ATTESTATION_RECORD.md`, and merges it into the `main` branch.
