"""
Gesture Streamer — WebSocket streaming for real-time avatar control.

Streams gesture data from the AI engine to Unity in real-time
using WebSocket connections for low-latency animation updates.
"""

import asyncio
import json
from fastapi import WebSocket, WebSocketDisconnect

from avatar_animation.gesture_mapper import GestureMapper


class GestureStreamer:

    def __init__(self):
        self.mapper = GestureMapper()
        self.connections = []

    async def connect(self, websocket: WebSocket):
        """Accept a new WebSocket connection."""
        await websocket.accept()
        self.connections.append(websocket)

    def disconnect(self, websocket: WebSocket):
        """Remove a disconnected client."""
        self.connections.remove(websocket)

    async def broadcast(self, message: dict):
        """Broadcast animation data to all connected Unity clients."""
        payload = json.dumps(message)
        disconnected = []

        for ws in self.connections:
            try:
                await ws.send_text(payload)
            except Exception:
                disconnected.append(ws)

        for ws in disconnected:
            self.connections.remove(ws)

    async def stream_gesture(self, gesture_token):
        """Convert gesture to animation and broadcast."""
        clips = self.mapper.map_token(gesture_token)

        message = {
            "type": "gesture",
            "token": gesture_token,
            "clips": clips if isinstance(clips, list) else [clips],
            "timestamp": asyncio.get_event_loop().time()
        }

        await self.broadcast(message)

    async def stream_sequence(self, tokens):
        """Stream a full sign sequence with timing."""
        for token in tokens:
            await self.stream_gesture(token)
            await asyncio.sleep(0.8)  # Animation duration


streamer = GestureStreamer()
