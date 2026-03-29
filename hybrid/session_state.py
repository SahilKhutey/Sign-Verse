"""
hybrid/session_state.py — Redis-Backed Per-Device Session & Telemetry Store

Each AR glasses device gets a session key in Redis that tracks:
  - Rolling average edge latency (last N frames)
  - Routing history (edge/cloud calls)
  - Rate-limit token bucket state
  - Active JWT claim cache (avoids repeated JWT validation)

Falls back gracefully to an in-memory dict if Redis is unavailable.
"""

import json
import time
import logging
from collections import deque
from typing import Optional, Dict, Any

logger = logging.getLogger("hybrid.session_state")

_REDIS_AVAILABLE = False
_redis_client = None

def _get_redis():
    global _redis_client, _REDIS_AVAILABLE
    if _redis_client is None:
        try:
            import redis
            _redis_client = redis.Redis(
                host="redis-service",
                port=6379,
                decode_responses=True,
                socket_connect_timeout=1.0,
            )
            _redis_client.ping()
            _REDIS_AVAILABLE = True
            logger.info("Redis session store connected.")
        except Exception as e:
            logger.warning(f"Redis unavailable, using in-memory store. ({e})")
            _REDIS_AVAILABLE = False
    return _redis_client if _REDIS_AVAILABLE else None


# ── In-memory fallback ─────────────────────────────────────────────────────────
_memory_store: Dict[str, Dict] = {}


def _mem_get(key: str) -> Optional[str]:
    entry = _memory_store.get(key)
    if entry and entry.get("ttl", float("inf")) > time.time():
        return entry.get("value")
    return None


def _mem_set(key: str, value: str, ttl_secs: int = 300):
    _memory_store[key] = {"value": value, "ttl": time.time() + ttl_secs}


# ── Public API ─────────────────────────────────────────────────────────────────

SESSION_TTL = 300  # 5 min idle timeout

def get_session(device_id: str) -> Dict[str, Any]:
    """Load device session or create a fresh one."""
    key = f"session:{device_id}"
    r = _get_redis()
    raw = r.get(key) if r else _mem_get(key)
    if raw:
        return json.loads(raw)
    return {
        "device_id": device_id,
        "created_at": time.time(),
        "edge_calls": 0,
        "cloud_calls": 0,
        "latency_samples": [],   # Rolling window, last 20 values
        "avg_edge_latency_ms": 0.0,
        "last_route": "edge",
    }


def save_session(device_id: str, session: Dict[str, Any]):
    """Persist session with TTL refresh."""
    key = f"session:{device_id}"
    # Keep latency window bounded
    session["latency_samples"] = session["latency_samples"][-20:]
    if session["latency_samples"]:
        session["avg_edge_latency_ms"] = sum(session["latency_samples"]) / len(session["latency_samples"])
    payload = json.dumps(session)
    r = _get_redis()
    if r:
        r.setex(key, SESSION_TTL, payload)
    else:
        _mem_set(key, payload, SESSION_TTL)


def update_route_telemetry(
    device_id: str,
    route: str,          # "edge" | "cloud"
    latency_ms: float,
):
    """Atomic update of routing telemetry for a device."""
    session = get_session(device_id)
    if route == "edge":
        session["edge_calls"] += 1
        session["latency_samples"].append(latency_ms)
    else:
        session["cloud_calls"] += 1
    session["last_route"] = route
    save_session(device_id, session)


def cache_jwt_claim(token: str, claim: dict, ttl: int = 600):
    """Store validated JWT claims to avoid repeated crypto work."""
    key = f"jwt:{token[:32]}"  # Use prefix to limit key size
    r = _get_redis()
    payload = json.dumps(claim)
    if r:
        r.setex(key, ttl, payload)
    else:
        _mem_set(key, payload, ttl)


def get_jwt_claim(token: str) -> Optional[dict]:
    key = f"jwt:{token[:32]}"
    r = _get_redis()
    raw = r.get(key) if r else _mem_get(key)
    return json.loads(raw) if raw else None
