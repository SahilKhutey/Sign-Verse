from fastapi import FastAPI, UploadFile, HTTPException
from ai_engine.inference_pipeline import InferencePipeline
import shutil
import os

app = FastAPI()

pipeline = InferencePipeline()

# Ensure temp directory exists
os.makedirs("temp", exist_ok=True)


@app.post("/speech-to-sign/")
async def speech_to_sign(file: UploadFile):
    """
    POST /speech-to-sign/

    Accepts an audio file upload and returns:
    {
        "text": "hello how are you",
        "sign_tokens": ["HELLO", "HOW", "YOU"]
    }

    These tokens drive the 3D avatar animations in Unity.
    """

    if file is None or not file.filename:
        raise HTTPException(status_code=400, detail="Missing audio file")

    file_location = f"temp/{file.filename}"

    with open(file_location, "wb") as buffer:
        shutil.copyfileobj(file.file, buffer)

    result = pipeline.speech_to_sign(file_location)

    return result
