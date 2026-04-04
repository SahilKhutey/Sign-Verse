"""
Animation API — REST API bridge between AI engine and Unity.

Provides endpoints for Unity to fetch animation sequences,
query current state, and control playback.
"""

from fastapi import FastAPI
from pydantic import BaseModel
from typing import List

from avatar_animation.gesture_mapper import GestureMapper
from avatar_animation.animation_controller import AnimationController

app = FastAPI(title="SignVerse Avatar Animation API")

mapper = GestureMapper()
controller = AnimationController()


class SignTokens(BaseModel):
    tokens: List[str]


class SpeedRequest(BaseModel):
    speed: float


@app.post("/animate")
async def animate_tokens(request: SignTokens):
    """Convert sign tokens to animation sequence and queue."""
    clips = mapper.map_sequence(request.tokens)
    controller.enqueue(clips)
    return {
        "clips": clips,
        "queued": len(clips)
    }


@app.get("/next")
async def next_animation():
    """Get next animation clip to play."""
    clip = controller.play_next()
    return {"animation": clip}


@app.get("/state")
async def get_state():
    """Get current animation controller state."""
    return controller.get_state()


@app.post("/speed")
async def set_speed(request: SpeedRequest):
    """Control animation playback speed."""
    controller.set_speed(request.speed)
    return {"speed": controller.playback_speed}


@app.post("/clear")
async def clear_queue():
    """Clear all queued animations."""
    controller.clear()
    return {"status": "cleared"}
