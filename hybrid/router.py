"""
hybrid/router.py — Smart Latency-Aware Routing Decision Engine

Decides whether each inference request should run locally on the AR device
or be offloaded to cloud AI.

Routing Factors (weighted score):
  - Edge confidence score        (primary signal)
  - Rolling average edge latency (self-reported by device)
  - Sequence complexity          (token entropy / gesture count)
  - Network RTT to cloud         (ping telemetry)
  - Cloud pod availability       (from load balancer)
"""

import time
import math
import logging
from dataclasses import dataclass, field
from typing import Optional

logger = logging.getLogger("hybrid.router")

# ── Constants ──────────────────────────────────────────────────────────────────

CONFIDENCE_THRESHOLD    = 0.50   # Below this → prefer cloud
LATENCY_BUDGET_MS       = 45.0   # Edge must stay under this
COMPLEXITY_THRESHOLD    = 0.65   # Entropy score above this → cloud
NETWORK_RTT_MAX_MS      = 80.0   # If RTT > this, stick with edge
CLOUD_UNAVAILABLE_SCORE = 999.0  # Sentinel for no cloud pods


@dataclass
class EdgeTelemetry:
    """Real-time metrics reported by the AR glasses each frame."""
    device_id: str
    confidence: float            # Edge model output confidence [0-1]
    edge_latency_ms: float       # Most recent local inference time
    sequence_entropy: float      # Token distribution entropy [0-1]
    gesture_complexity: int      # Number of distinct tokens in last 64-frame window
    network_rtt_ms: float        # Round-trip to cloud (ms), 0 if unmeasured
    timestamp: float = field(default_factory=time.time)


@dataclass
class RoutingDecision:
    mode: str           # "edge" | "cloud"
    score: float        # Raw routing score (lower = prefer edge)
    reason: str
    ttl_ms: int = 500   # How long this decision is valid before re-evaluation


class RoutingEngine:
    """
    Stateless, low-overhead routing engine.
    Each call to `decide()` returns a routing decision in <1ms.
    """

    def __init__(
        self,
        confidence_threshold: float = CONFIDENCE_THRESHOLD,
        latency_budget_ms: float = LATENCY_BUDGET_MS,
        complexity_threshold: float = COMPLEXITY_THRESHOLD,
        network_rtt_max_ms: float = NETWORK_RTT_MAX_MS,
    ):
        self.confidence_threshold = confidence_threshold
        self.latency_budget_ms = latency_budget_ms
        self.complexity_threshold = complexity_threshold
        self.network_rtt_max_ms = network_rtt_max_ms

    def decide(
        self,
        telemetry: EdgeTelemetry,
        cloud_available: bool = True,
    ) -> RoutingDecision:
        """
        Compute a weighted routing score and return a decision.

        SCORE COMPONENTS (each 0–1, lower = stay on edge):
          - confidence_penalty:   1 - confidence
          - latency_penalty:      latency / budget
          - complexity_penalty:   sequence_entropy
          - network_penalty:      rtt / rtt_max  (penalty for going to cloud)
        """
        if not cloud_available:
            return RoutingDecision(
                mode="edge",
                score=0.0,
                reason="cloud_unavailable",
            )

        # Individual factor scores
        conf_penalty    = 1.0 - max(0.0, min(1.0, telemetry.confidence))
        latency_penalty = min(1.0, telemetry.edge_latency_ms / self.latency_budget_ms)
        entropy_penalty = max(0.0, min(1.0, telemetry.sequence_entropy))
        # Network penalty discourages cloud when RTT is high
        rtt_penalty = min(1.0, telemetry.network_rtt_ms / self.network_rtt_max_ms) \
                      if telemetry.network_rtt_ms > 0 else 0.3

        # Weighted composite score (higher → prefer cloud)
        score = (
            0.40 * conf_penalty +
            0.25 * latency_penalty +
            0.20 * entropy_penalty +
            0.15 * -rtt_penalty   # Negative: high RTT discourages cloud
        )

        # --- Decision gates -----------------------------------------------
        # Hard gate 1: edge is very confident and fast → always stay local
        if (
            telemetry.confidence >= self.confidence_threshold
            and telemetry.edge_latency_ms < self.latency_budget_ms
            and telemetry.sequence_entropy < self.complexity_threshold
        ):
            return RoutingDecision(mode="edge", score=score, reason="edge_sufficient")

        # Hard gate 2: RTT too high, cloud would be slower than edge
        if telemetry.network_rtt_ms > self.network_rtt_max_ms:
            return RoutingDecision(
                mode="edge", score=score, reason="high_rtt_stay_local"
            )

        # Soft gate: use score threshold for routing
        if score > 0.35:
            return RoutingDecision(mode="cloud", score=score, reason="low_confidence_or_complex")

        return RoutingDecision(mode="edge", score=score, reason="score_below_threshold")
