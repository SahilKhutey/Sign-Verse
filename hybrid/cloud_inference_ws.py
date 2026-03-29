"""
hybrid/cloud_inference_ws.py — Dedicated WebSocket Endpoint for Cloud Fallback

This FastAPI router mounts at /ws/hybrid and handles:
  1. JWT authentication via token query param
  2. Per-device rate limiting
  3. Smart routing decision (edge telemetry → decide edge/cloud)
  4. Cloud inference via the load balancer
  5. Session telemetry updates via Redis

Protocol:
  Client sends JSON:
  {
    "type":        "infer",
    "device_id":   "ar-glasses-001",
    "keypoints":   [848-dim float array],
    "confidence":  0.31,
    "latency_ms":  48.2,
    "entropy":     0.78,
    "network_rtt": 32.0,
    "frame_id":    1234567890
  }

  Server responds:
  {
    "type":        "result",
    "route":       "edge" | "cloud",
    "reason":      "...",
    "gesture_id":  42,
    "gesture_label": "HELLO",
    "confidence":  0.91,
    "latency_ms":  15,
    "frame_id":    1234567890
  }
"""

import os
import time
import logging
import asyncio
from typing import List

from fastapi import APIRouter, WebSocket, WebSocketDisconnect

from hybrid.router        import RoutingEngine, EdgeTelemetry
from hybrid.session_state import update_route_telemetry, get_jwt_claim, cache_jwt_claim
from hybrid.rate_limiter  import HybridRateLimiter
from hybrid.load_balancer import HybridLoadBalancer

logger = logging.getLogger("hybrid.cloud_ws")

router = APIRouter()

# ── Singletons ─────────────────────────────────────────────────────────────────
_routing_engine = RoutingEngine()
_rate_limiter   = HybridRateLimiter()

# Cloud pod URLs — populated from ENV or K8s service discovery
_POD_URLS = os.getenv(
    "CLOUD_POD_URLS",
    "http://signverse-inference-service:8000"
).split(",")
_load_balancer = HybridLoadBalancer(_POD_URLS)


def _verify_token(token: str) -> bool:
    """JWT verification with Redis claim cache."""
    if not token:
        return False
    cached = get_jwt_claim(token)
    if cached:
        return True
    # TODO: Replace with real JWT validation (e.g. python-jose)
    stream_token = os.getenv("STREAM_TOKEN", "")
    if stream_token and token == stream_token:
        cache_jwt_claim(token, {"valid": True})
        return True
    return False


@router.websocket("/ws/hybrid")
async def hybrid_websocket(ws: WebSocket):
    """Main hybrid inference WebSocket — latency-aware edge/cloud routing."""
    token = ws.query_params.get("token", "")
    if not _verify_token(token):
        await ws.close(code=1008, reason="Invalid or missing token")
        return

    await ws.accept()
    logger.info("Hybrid WS client connected.")

    try:
        while True:
            data = await ws.receive_json()
            frame_start = time.perf_counter()

            msg_type  = data.get("type", "infer")
            device_id = data.get("device_id", "unknown")
            frame_id  = data.get("frame_id", 0)

            # ── Rate limit check ───────────────────────────────────────────
            allowed, rl_reason = _rate_limiter.is_allowed(device_id)
            if not allowed:
                await ws.send_json({
                    "type":    "error",
                    "code":    "rate_limited",
                    "detail":  rl_reason,
                    "frame_id": frame_id,
                })
                continue

            # ── Build telemetry struct ─────────────────────────────────────
            telemetry = EdgeTelemetry(
                device_id        = device_id,
                confidence       = float(data.get("confidence",      0.0)),
                edge_latency_ms  = float(data.get("latency_ms",      50.0)),
                sequence_entropy = float(data.get("entropy",         0.5)),
                gesture_complexity = int(data.get("gesture_complexity", 5)),
                network_rtt_ms   = float(data.get("network_rtt",     0.0)),
            )

            # ── Routing decision ───────────────────────────────────────────
            decision = _routing_engine.decide(telemetry, cloud_available=True)
            route = decision.mode

            result_payload: dict = {"type": "result", "frame_id": frame_id, "route": route, "reason": decision.reason}

            if route == "edge":
                # Tell client to keep using its local result
                result_payload.update({
                    "confidence": telemetry.confidence,
                    "latency_ms": int((time.perf_counter() - frame_start) * 1000),
                })
            else:
                # ── Cloud inference via load balancer ──────────────────────
                keypoints = data.get("keypoints", [])
                if not keypoints:
                    await ws.send_json({"type": "error", "code": "no_keypoints", "frame_id": frame_id})
                    continue
                try:
                    cloud_result = await _load_balancer.post_json(
                        "/gesture/classify-sequence",
                        {"sequence": [keypoints]},
                        timeout=3.0,
                    )
                    result_payload.update({
                        "gesture_id":    cloud_result.get("gesture_id"),
                        "gesture_label": cloud_result.get("gesture_label"),
                        "confidence":    cloud_result.get("confidence"),
                        "latency_ms":    int((time.perf_counter() - frame_start) * 1000),
                    })
                except Exception as e:
                    # Graceful fallback: tell client to use edge result
                    logger.warning(f"Cloud inference failed, falling back to edge. {e}")
                    result_payload.update({
                        "route":  "edge",
                        "reason": "cloud_error_fallback",
                        "confidence": telemetry.confidence,
                        "latency_ms": int((time.perf_counter() - frame_start) * 1000),
                    })

            # ── Update session telemetry async (non-blocking) ──────────────
            asyncio.get_event_loop().call_soon(
                update_route_telemetry,
                device_id,
                result_payload["route"],
                float(result_payload.get("latency_ms", 0)),
            )

            await ws.send_json(result_payload)

    except WebSocketDisconnect:
        logger.info(f"Hybrid WS client disconnected.")
    except Exception as e:
        logger.exception(f"Hybrid WS error: {e}")
        await ws.close(code=1011, reason="Internal server error")
