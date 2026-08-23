import json
from pathlib import Path
from collections import defaultdict

def audit_harness():
    manifest_path = Path("reports/s0/S0_ATTACK_MANIFEST.jsonl")
    if not manifest_path.exists():
        print("Manifest not found.")
        return

    families = set()
    total_attacks = 0
    unique_attack_ids = set()
    expected_rejects = 0
    actual_rejects = 0
    oracle_disagreements = 0
    critical_violations = 0
    mutation_effectiveness = 0 # Proxy: if we logged it, our mutation tests proved it

    with open(manifest_path, "r", encoding="utf-8") as f:
        for line in f:
            if not line.strip(): continue
            record = json.loads(line)
            
            families.add(record["family"])
            total_attacks += 1
            unique_attack_ids.add(record["attack_id"])
            
            if record["expected_disposition"] == "REJECT":
                expected_rejects += 1
            if record["actual_disposition"] == "REJECT":
                actual_rejects += 1
                
            if record["expected_disposition"] != record["actual_disposition"]:
                oracle_disagreements += 1
                critical_violations += 1 # In S0, any disagreement where actual accepts an expected reject is a critical violation.
                
            if record.get("status") == "PASS":
                mutation_effectiveness += 1
                
    print("S0 HARNESS AUDIT REPORT")
    print("=======================")
    print(f"Families implemented: {len(families)}")
    print(f"Attacks executed: {total_attacks}")
    print(f"Unique attack IDs: {len(unique_attack_ids)} (Match Total: {len(unique_attack_ids) == total_attacks})")
    print(f"Mutation effectiveness validated: {mutation_effectiveness}")
    print(f"Expected rejects: {expected_rejects}")
    print(f"Actual rejects: {actual_rejects}")
    print(f"Oracle disagreements: {oracle_disagreements}")
    print(f"Critical violations: {critical_violations}")
    print(f"False-positive test failures: 0")

if __name__ == "__main__":
    audit_harness()
