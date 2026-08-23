import sys
import hashlib
from pathlib import Path
from cryptography.hazmat.primitives.asymmetric import padding
from cryptography.hazmat.primitives import hashes, serialization
from cryptography.exceptions import InvalidSignature

def verify_report(report_path: str, digest_path: str, sig_path: str, pub_key_path: str):
    rp = Path(report_path)
    dp = Path(digest_path)
    sp = Path(sig_path)
    pkp = Path(pub_key_path)
    
    if not all(p.exists() for p in [rp, dp, sp, pkp]):
        print("Missing required artifacts for verification.")
        sys.exit(1)
        
    with open(rp, "r", encoding="utf-8") as f:
        report_content = f.read()
        
    with open(dp, "r", encoding="utf-8") as f:
        claimed_digest = f.read().strip()
        
    with open(sp, "rb") as f:
        signature = f.read()
        
    with open(pkp, "rb") as f:
        public_key = serialization.load_pem_public_key(f.read())
        
    # 1. Verify Digest Matches Report
    hasher = hashlib.sha256()
    hasher.update(report_content.encode('utf-8'))
    actual_digest = hasher.hexdigest()
    
    if actual_digest != claimed_digest:
        print("[!] REJECT: Report content does not match the claimed SHA-256 digest.")
        sys.exit(1)
        
    # 2. Verify Signature Matches Digest
    try:
        public_key.verify(
            signature,
            claimed_digest.encode('utf-8'),
            padding.PSS(mgf=padding.MGF1(hashes.SHA256()), salt_length=padding.PSS.MAX_LENGTH),
            hashes.SHA256()
        )
        print("[+] PASS: Cryptographic attestation valid. The report is verified by the trusted key.")
    except InvalidSignature:
        print("[!] REJECT: Cryptographic signature is INVALID for the provided digest.")
        sys.exit(1)

if __name__ == "__main__":
    if len(sys.argv) < 5:
        print("Usage: python verify_s0_report.py <report.md> <digest.sha256> <signature.sig> <public_key.pub>")
        sys.exit(1)
    verify_report(sys.argv[1], sys.argv[2], sys.argv[3], sys.argv[4])
