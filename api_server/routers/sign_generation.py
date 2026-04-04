from fastapi import APIRouter

router = APIRouter(prefix="/generate", tags=["generation"])

@router.post("/motion")
async def generate_motion():
    return {"status": "success", "motion_id": "M1"}
