import sys
from pathlib import Path
from cryptography.hazmat.primitives.asymmetric import rsa, padding
from cryptography.hazmat.primitives import hashes, serialization

def sign_report(digest_path: str, key_path: str):
    dp = Path(digest_path)
    if not dp.exists():
        print(f"Digest not found at {digest_path}")
        sys.exit(1)
        
    kp = Path(key_path)
    if not kp.exists():
        print("Generating new offline private key...")
        kp.parent.mkdir(parents=True, exist_ok=True)
        private_key = rsa.generate_private_key(public_exponent=65537, key_size=2048)
        
        with open(kp, "wb") as f:
            f.write(private_key.private_bytes(
                encoding=serialization.Encoding.PEM,
                format=serialization.PrivateFormat.PKCS8,
                encryption_algorithm=serialization.NoEncryption()
            ))
            
        public_key = private_key.public_key()
        with open(kp.with_suffix(".pub"), "wb") as f:
            f.write(public_key.public_bytes(
                encoding=serialization.Encoding.PEM,
                format=serialization.PublicFormat.SubjectPublicKeyInfo
            ))
    else:
        with open(kp, "rb") as f:
            private_key = serialization.load_pem_private_key(f.read(), password=None)
            
    with open(dp, "r", encoding="utf-8") as f:
        digest = f.read().strip()
        
    signature = private_key.sign(
        digest.encode('utf-8'),
        padding.PSS(mgf=padding.MGF1(hashes.SHA256()), salt_length=padding.PSS.MAX_LENGTH),
        hashes.SHA256()
    )
    
    sig_path = dp.with_suffix(".sig")
    with open(sig_path, "wb") as f:
        f.write(signature)
        
    print(f"Successfully signed {digest_path} -> {sig_path}")

if __name__ == "__main__":
    if len(sys.argv) < 3:
        print("Usage: python sign_s0_report.py <digest_file.sha256> <private_key.pem>")
        sys.exit(1)
    sign_report(sys.argv[1], sys.argv[2])
