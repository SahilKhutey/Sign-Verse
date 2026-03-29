"""
API Server Core Security — JWT & Encryption
Handles token validation and biometric data encryption.
"""

import os
import jwt
from datetime import datetime, timedelta
from typing import Optional
from fastapi import HTTPException, Security
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials
from cryptography.fernet import Fernet

# Configuration
SECRET_KEY = os.getenv("SECRET_KEY", "signverse_super_secret_key_10b_scale")
ALGORITHM = "HS256"
ACCESS_TOKEN_EXPIRE_MINUTES = 60
BIOMETRIC_ENCRYPTION_KEY = os.getenv("BIOMETRIC_ENCRYPTION_KEY", Fernet.generate_key().decode())

security = HTTPBearer()
fernet = Fernet(BIOMETRIC_ENCRYPTION_KEY.encode())

def create_access_token(data: dict, expires_delta: Optional[timedelta] = None):
    to_encode = data.copy()
    if expires_delta:
        expire = datetime.utcnow() + expires_delta
    else:
        expire = datetime.utcnow() + timedelta(minutes=ACCESS_TOKEN_EXPIRE_MINUTES)
    to_encode.update({"exp": expire})
    encoded_jwt = jwt.encode(to_encode, SECRET_KEY, algorithm=ALGORITHM)
    return encoded_jwt

async def verify_token(auth: HTTPAuthorizationCredentials = Security(security)):
    """Verifies JWT token for API access."""
    try:
        payload = jwt.decode(auth.credentials, SECRET_KEY, algorithms=[ALGORITHM])
        return payload
    except jwt.ExpiredSignatureError:
        raise HTTPException(status_code=401, detail="Token expired")
    except jwt.InvalidTokenError:
        raise HTTPException(status_code=401, detail="Invalid token")

def encrypt_biometrics(vector: list) -> str:
    """Encrypts high-fidelity keypoint vectors for privacy-safe storage."""
    import json
    data_str = json.dumps(vector)
    return fernet.encrypt(data_str.encode()).decode()

def decrypt_biometrics(encrypted_str: str) -> list:
    """Decrypts high-fidelity keypoint vectors."""
    import json
    decrypted_data = fernet.decrypt(encrypted_str.encode()).decode()
    return json.loads(decrypted_data)

if __name__ == "__main__":
    # Test encryption
    sample = [0.1, 0.2, 0.3]
    enc = encrypt_biometrics(sample)
    dec = decrypt_biometrics(enc)
    print(f"Encryption test: {sample == dec}")
