from fastapi import APIRouter, HTTPException, BackgroundTasks
from pydantic import BaseModel
from typing import List, Optional
import os
import json
import uuid
from datetime import datetime

router = APIRouter(prefix="/feedback", tags=["Feedback"])

FEEDBACK_DIR = "datasets/user_feedback/pending_validation"
os.makedirs(FEEDBACK_DIR, exist_ok=True)

class TranslationCorrection(BaseModel):
    original_text: str
    corrected_text: str
    language: str
    video_session_id: str
    sequence_data: List[List[float]]  # Keypoints/Frames
    timestamp: Optional[str] = None

@router.post("/correction")
async def submit_correction(correction: TranslationCorrection, background_tasks: BackgroundTasks):
    """
    Endpoint to receive translation corrections from users.
    Stores data for incremental fine-tuning.
    """
    try:
        feedback_id = str(uuid.uuid4())
        
        # Ensure timestamp is set
        if not correction.timestamp:
            correction.timestamp = datetime.now().isoformat()
            
        file_path = os.path.join(FEEDBACK_DIR, f"{feedback_id}.json")
        
        # In a real production system, this would go to a database or blob storage
        with open(file_path, "w") as f:
            json.dump(correction.dict(), f, indent=2)
            
        return {
            "status": "success", 
            "feedback_id": feedback_id, 
            "message": "Correction received and queued for validation."
        }
    except Exception as e:
        print(f"Error saving feedback: {e}")
        raise HTTPException(status_code=500, detail="Failed to save feedback.")

@router.get("/stats")
async def get_feedback_stats():
    """
    Returns statistics about collected feedback.
    """
    files = os.listdir(FEEDBACK_DIR)
    return {
        "pending_count": len(files),
        "directory": FEEDBACK_DIR
    }
