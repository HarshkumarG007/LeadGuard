import subprocess
import os
import sys
from pathlib import Path

def run_cmd(cmd):
    result = subprocess.run(cmd, shell=True, capture_output=True, text=True)
    return result.returncode, result.stdout, result.stderr

def run_hostile_test():
    print("=== HOSTILE MAINTAINER ATTACK SIMULATION ===")
    
    # 1. Clean slate
    os.makedirs("keys", exist_ok=True)
    private_key = "keys/human_offline_key.pem"
    public_key = "keys/human_offline_key.pub"
    
    # The external human generates the key and signs a valid report
    print("\n[Human] Generating offline key and signing the real report...")
    # Generate the initial S0 run (creates the report and digest)
    run_cmd(".venv\\Scripts\\python.exe scripts/run_s0_adversarial.py")
    
    report_file = "reports/s0/S0_ATTACK_REPORT.md"
    digest_file = "reports/s0/S0_ATTACK_REPORT.sha256"
    sig_file = "reports/s0/S0_ATTACK_REPORT.sig"
    
    run_cmd(f".venv\\Scripts\\python.exe scripts/sign_s0_report.py {digest_file} {private_key}")
    
    # Verify it passes normally
    code, out, err = run_cmd(f".venv\\Scripts\\python.exe scripts/verify_s0_report.py {report_file} {digest_file} {sig_file} {public_key}")
    if code != 0:
        print("Initial baseline failed! Test broken.")
        sys.exit(1)
        
    print("[+] Baseline verification passed.")
    
    # 2. Hostile Maintainer Attack
    print("\n[Hostile Maintainer] Modifying the report to falsely claim passing status...")
    with open(report_file, "r", encoding="utf-8") as f:
        content = f.read()
    
    content = content.replace("CONDITIONAL_GO", "PASS")
    content = content.replace("INCOMPLETE", "COMPLETE")
    
    with open(report_file, "w", encoding="utf-8") as f:
        f.write(content)
        
    print("[Hostile Maintainer] Recomputing SHA-256 to match the tampered report...")
    run_cmd(f".venv\\Scripts\\python.exe scripts/run_s0_adversarial.py") # Will overwrite digest to match the bad report
    
    # Simulate that they CANNOT regenerate the sig because the offline key is offline
    print("[Hostile Maintainer] Attempting to pass the tampered artifacts to the independent verifier...")
    code, out, err = run_cmd(f".venv\\Scripts\\python.exe scripts/verify_s0_report.py {report_file} {digest_file} {sig_file} {public_key}")
    
    if "REJECT: Cryptographic signature is INVALID" in out:
        print("[+] SUCCESS: Independent verifier successfully rejected the hostile maintainer's tampered report.")
        print("Output snippet:\n", out.strip())
    else:
        print("[-] FAILURE: Independent verifier was fooled! Output:\n", out)
        sys.exit(1)

if __name__ == "__main__":
    run_hostile_test()
