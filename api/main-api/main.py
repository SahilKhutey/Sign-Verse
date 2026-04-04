from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.middleware.cors import CORSMiddleware
import logging
import json
import asyncio

# Sign-Verse Main API
app = FastAPI(title="Sign-Verse Main API")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

logger = logging.getLogger("Sign-Verse.MainAPI")

class ConnectionManager:
    """
    Manages real-time WebSocket connections for the Sign-Verse Dashboard.
    """
    def __init__(self):
        self.active_connections: list[WebSocket] = []

    async def connect(self, websocket: WebSocket):
        await websocket.accept()
        self.active_connections.append(websocket)

    def disconnect(self, websocket: WebSocket):
        self.active_connections.remove(websocket)

    async def broadcast(self, message: str):
        for connection in self.active_connections:
            await connection.send_text(message)

manager = ConnectionManager()

@app.get("/")
async def root():
    return {"message": "Sign-Verse Main API v1.0 ONLINE"}

@app.websocket("/ws/telemetry")
async def websocket_endpoint(websocket: WebSocket):
    """
    Real-time telemetry stream for 3D landmarks and hardware status.
    """
    await manager.connect(websocket)
    try:
        while True:
            # Placeholder for data streaming from Processing Layer
            data = await websocket.receive_text()
            await manager.broadcast(f"Telemetry Update: {data}")
    except WebSocketDisconnect:
        manager.disconnect(websocket)
        logger.info("Client disconnected from telemetry stream")

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8080)
