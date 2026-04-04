"""
hybrid/load_balancer.py — Weighted Round-Robin Load Balancer

Routes cloud inference requests across available inference pods.
Tracks pod health, latency, and error rates.
Integrates with Kubernetes via the K8s service DNS (round-robin is
handled by kube-proxy, but this layer adds latency-aware affinity).
"""

import time
import asyncio
import logging
import httpx
from dataclasses import dataclass, field
from typing import List, Optional

logger = logging.getLogger("hybrid.load_balancer")


@dataclass
class InferencePod:
    """Represents one cloud inference backend pod."""
    url: str
    weight: int = 10
    active_connections: int = 0
    avg_latency_ms: float = 50.0
    error_count: int = 0
    healthy: bool = True
    last_health_check: float = field(default_factory=time.time)

    @property
    def effective_weight(self) -> float:
        """Lower latency + fewer errors = higher effective weight."""
        if not self.healthy:
            return 0.0
        latency_factor = max(0.1, 1.0 - (self.avg_latency_ms / 500.0))
        error_factor   = max(0.1, 1.0 - (self.error_count / 100.0))
        return self.weight * latency_factor * error_factor


class HybridLoadBalancer:
    """
    Weighted least-connections load balancer for cloud inference pods.
    
    Selection strategy:
        Score = effective_weight / (active_connections + 1)
        → Pod with highest score gets the request.
    """

    def __init__(self, pod_urls: List[str]):
        self.pods = [InferencePod(url=url) for url in pod_urls]
        self._lock = asyncio.Lock()

    def select_pod(self) -> Optional[InferencePod]:
        """Select the best pod using weighted least-connections."""
        healthy = [p for p in self.pods if p.healthy]
        if not healthy:
            return None
        return max(healthy, key=lambda p: p.effective_weight / (p.active_connections + 1))

    async def post_json(self, path: str, payload: dict, timeout: float = 3.0) -> dict:
        """Route a request to the best pod and track latency."""
        pod = self.select_pod()
        if not pod:
            raise RuntimeError("No healthy cloud inference pods available.")

        async with self._lock:
            pod.active_connections += 1

        url = pod.url.rstrip("/") + path
        start = time.perf_counter()
        try:
            async with httpx.AsyncClient(timeout=timeout) as client:
                resp = await client.post(url, json=payload)
                resp.raise_for_status()
                latency_ms = (time.perf_counter() - start) * 1000
                pod.avg_latency_ms = 0.8 * pod.avg_latency_ms + 0.2 * latency_ms
                return resp.json()
        except Exception as e:
            pod.error_count += 1
            logger.error(f"Pod {pod.url} failed: {e}")
            raise
        finally:
            async with self._lock:
                pod.active_connections = max(0, pod.active_connections - 1)

    async def health_check_loop(self, interval_secs: float = 10.0):
        """Background coroutine to probe pod health."""
        while True:
            for pod in self.pods:
                try:
                    async with httpx.AsyncClient(timeout=2.0) as client:
                        resp = await client.get(pod.url.rstrip("/") + "/health")
                        pod.healthy = resp.status_code == 200
                        pod.error_count = max(0, pod.error_count - 1)  # Decay errors
                except Exception:
                    pod.healthy = False
                    logger.warning(f"Pod {pod.url} is unhealthy.")
                pod.last_health_check = time.time()
            await asyncio.sleep(interval_secs)
