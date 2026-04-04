from fastapi import APIRouter

router = APIRouter(prefix="/auth", tags=["auth"])

@router.post("/login")
async def login():
    return {"status": "authenticated", "token": "test_token"}
