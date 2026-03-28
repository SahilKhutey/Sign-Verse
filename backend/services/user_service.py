"""
User Service — Registration, login, and profile management.
"""

from fastapi import APIRouter, Depends, HTTPException, Header
from pydantic import BaseModel
from sqlalchemy.orm import Session
from datetime import datetime, timedelta
from database.db_connection import get_db
from database.models import User, RefreshToken
from utils.authentication import (
    hash_password,
    verify_password,
    create_token,
    create_refresh_token,
    verify_refresh_token,
    hash_token,
    get_current_user,
)
from utils.rate_limit import limiter

router = APIRouter()


class UserCreate(BaseModel):
    username: str
    email: str
    password: str


class UserLogin(BaseModel):
    email: str
    password: str


class RefreshRequest(BaseModel):
    refresh_token: str


async def verify_auth(authorization: str = Header(None)):
    if not authorization or not authorization.startswith("Bearer "):
        raise HTTPException(status_code=401, detail="Missing or invalid token")
    token = authorization.split(" ")[1]
    try:
        return get_current_user(token)
    except Exception as e:
        raise HTTPException(status_code=401, detail=str(e))


@router.post("/register")
@limiter.limit("10/minute")
async def register(user: UserCreate, db: Session = Depends(get_db)):
    existing = db.query(User).filter(User.email == user.email).first()
    if existing:
        raise HTTPException(status_code=400, detail="Email already registered")

    new_user = User(
        username=user.username,
        email=user.email,
        hashed_password=hash_password(user.password)
    )
    db.add(new_user)
    db.commit()
    db.refresh(new_user)

    return {"id": new_user.id, "username": new_user.username}


@router.post("/login")
@limiter.limit("10/minute")
async def login(user: UserLogin, db: Session = Depends(get_db)):
    db_user = db.query(User).filter(User.email == user.email).first()

    if not db_user or not verify_password(user.password, db_user.hashed_password):
        raise HTTPException(status_code=401, detail="Invalid credentials")

    access_token = create_token({"user_id": db_user.id})
    refresh_token = create_refresh_token({"user_id": db_user.id})
    refresh_hash = hash_token(refresh_token)

    db_token = RefreshToken(
        user_id=db_user.id,
        token_hash=refresh_hash,
        expires_at=datetime.utcnow() + timedelta(days=14),
        revoked=False,
    )
    db.add(db_token)
    db.commit()

    return {
        "token": access_token,
        "refresh_token": refresh_token,
        "username": db_user.username,
    }


@router.post("/refresh")
@limiter.limit("20/minute")
async def refresh(request: RefreshRequest, db: Session = Depends(get_db)):
    payload = verify_refresh_token(request.refresh_token)
    if not payload:
        raise HTTPException(status_code=401, detail="Invalid refresh token")

    token_hash = hash_token(request.refresh_token)
    stored = db.query(RefreshToken).filter(RefreshToken.token_hash == token_hash).first()
    if not stored or stored.revoked:
        raise HTTPException(status_code=401, detail="Refresh token revoked")

    if stored.expires_at < datetime.utcnow():
        raise HTTPException(status_code=401, detail="Refresh token expired")

    access_token = create_token({"user_id": payload.get("user_id")})
    return {"token": access_token}


@router.post("/logout")
@limiter.limit("20/minute")
async def logout(request: RefreshRequest, db: Session = Depends(get_db)):
    token_hash = hash_token(request.refresh_token)
    stored = db.query(RefreshToken).filter(RefreshToken.token_hash == token_hash).first()
    if stored:
        stored.revoked = True
        db.commit()
    return {"status": "ok"}


@router.get("/me")
async def get_me(db: Session = Depends(get_db), current_user: dict = Depends(verify_auth)):
    user_id = current_user.get("user_id")
    user = db.query(User).filter(User.id == user_id).first()
    if not user:
        raise HTTPException(status_code=404, detail="User not found")
    return {"id": user.id, "username": user.username, "email": user.email}


@router.get("/{user_id}")
async def get_profile(user_id: int, db: Session = Depends(get_db), current_user: dict = Depends(verify_auth)):
    if current_user.get("user_id") != user_id:
        raise HTTPException(status_code=403, detail="Not authorized")
    user = db.query(User).filter(User.id == user_id).first()
    if not user:
        raise HTTPException(status_code=404, detail="User not found")
    return {"id": user.id, "username": user.username, "email": user.email}
