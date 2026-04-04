from fastapi import FastAPI, WebSocket, WebSocketDisconnect, Request
from pydantic import BaseModel
import asyncio
import json
import time
import os
import sys

# Add root to sys.path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from core.translation_engine import SignVerseTranslationEngine

app = FastAPI(title="SignVerse Real-Time API")

class TextToSignRequest(BaseModel):
    text: str
    style: str = "neutral"
    output_format: str = "tokens"

class ConnectionManager:
    def __init__(self):
        self.active_connections = {}
        self.engine = SignVerseTranslationEngine()
    
    async def connect(self, websocket: WebSocket, user_id: str):
        await websocket.accept()
        self.active_connections[user_id] = websocket
        
        # Initialize session
        session = await self.engine.start_conversation_session(user_id)
        return session
    
    async def handle_video_stream(self, websocket: WebSocket, user_id: str):
        """Handle incoming video stream for sign to speech"""
        session = await self.connect(websocket, user_id)
        
        try:
            while True:
                # Receive video frame
                data = await websocket.receive_bytes()
                
                # Process in real-time
                result = await self.engine.process_frame(session, data)
                
                # Send back translation
                await websocket.send_json({
                    "type": "translation",
                    "text": result["text"],
                    "audio": result["speech"].tolist(),  # Convert to list for JSON
                    "timestamp": time.time()
                })
                
        except WebSocketDisconnect:
            if user_id in self.active_connections:
                del self.active_connections[user_id]
        except Exception as e:
            print(f"Error in video stream: {e}")
            if user_id in self.active_connections:
                del self.active_connections[user_id]

    async def handle_audio_stream(self, websocket: WebSocket, user_id: str):
        """Handle incoming audio stream for speech up sign"""
        await websocket.accept()
        self.active_connections[user_id] = websocket
        
        try:
            while True:
                data = await websocket.receive_bytes()
                result = await self.engine.speech_to_sign(data)
                
                await websocket.send_json({
                    "type": "gesture_tokens",
                    "text": result["text"],
                    "tokens": result["tokens"],
                    "timestamp": time.time()
                })
        except WebSocketDisconnect:
            if user_id in self.active_connections:
                del self.active_connections[user_id]

manager = ConnectionManager()

@app.websocket("/ws/translate/sign-to-speech")
async def sign_to_speech_endpoint(websocket: WebSocket):
    user_id = websocket.headers.get("X-User-ID", "anonymous")
    await manager.handle_video_stream(websocket, user_id)

@app.websocket("/ws/translate/speech-to-sign")
async def speech_to_sign_endpoint(websocket: WebSocket):
    user_id = websocket.headers.get("X-User-ID", "anonymous")
    await manager.handle_audio_stream(websocket, user_id)

@app.post("/api/translate/text-to-sign")
async def text_to_sign_endpoint(request: TextToSignRequest):
    """Batch text to sign translation"""
    engine = SignVerseTranslationEngine()
    
    result = await engine.text_to_sign.generate(
        request.text,
        style=request.style,
        output_format=request.output_format
    )
    
    return result

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8002)
