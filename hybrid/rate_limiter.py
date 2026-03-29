"""
hybrid/rate_limiter.py — Token Bucket + Sliding Window Rate Limiter

Provides per-device rate limiting for cloud inference offloads.
Uses Redis atomic LUA scripts for correctness under parallel connections.
Falls back to in-memory token buckets if Redis is unavailable.
"""

import time
import logging
from typing import Tuple

logger = logging.getLogger("hybrid.rate_limiter")

# Default limits
CLOUD_REQUESTS_PER_SECOND = 10     # Max cloud offloads per device per second
CLOUD_BURST_CAPACITY       = 30    # Token bucket burst size
GLOBAL_RPS_LIMIT           = 500   # Cluster-wide requests per second cap


# ── Redis LUA Script (atomic increment + expiry) ───────────────────────────────
_SLIDING_WINDOW_SCRIPT = """
local key     = KEYS[1]
local limit   = tonumber(ARGV[1])
local window  = tonumber(ARGV[2])
local now     = tonumber(ARGV[3])

redis.call('ZREMRANGEBYSCORE', key, 0, now - window)
local count = redis.call('ZCOUNT', key, '-inf', '+inf')
if count < limit then
    redis.call('ZADD', key, now, now)
    redis.call('EXPIRE', key, window)
    return 1  -- allowed
else
    return 0  -- denied
end
"""

# ── In-memory token bucket (fallback) ──────────────────────────────────────────
class _TokenBucket:
    def __init__(self, rate: float, capacity: int):
        self._rate     = rate        # Tokens per second
        self._capacity = capacity
        self._tokens   = float(capacity)
        self._last     = time.monotonic()

    def consume(self) -> bool:
        now = time.monotonic()
        elapsed = now - self._last
        self._tokens = min(self._capacity, self._tokens + elapsed * self._rate)
        self._last = now
        if self._tokens >= 1.0:
            self._tokens -= 1.0
            return True
        return False


_buckets: dict[str, _TokenBucket] = {}


class HybridRateLimiter:
    """
    Per-device rate limiter for cloud inference offloads.
    Uses Redis sliding window when available, token buckets otherwise.
    """

    def __init__(
        self,
        requests_per_second: int = CLOUD_REQUESTS_PER_SECOND,
        burst: int = CLOUD_BURST_CAPACITY,
    ):
        self._rps    = requests_per_second
        self._burst  = burst
        self._script = None
        self._r      = None
        self._init_redis()

    def _init_redis(self):
        try:
            import redis
            r = redis.Redis(host="redis-service", port=6379,
                           decode_responses=True, socket_connect_timeout=1.0)
            r.ping()
            self._r = r
            self._script = r.register_script(_SLIDING_WINDOW_SCRIPT)
            logger.info("Rate limiter using Redis sliding window.")
        except Exception as e:
            logger.warning(f"Rate limiter falling back to token bucket. ({e})")

    def is_allowed(self, device_id: str) -> Tuple[bool, str]:
        """
        Returns (allowed: bool, reason: str).
        Thread-safe when using Redis.
        """
        key = f"rl:{device_id}"
        now_ms = int(time.time() * 1000)
        window_ms = 1000  # 1-second sliding window

        if self._script and self._r:
            try:
                result = self._script(
                    keys=[key],
                    args=[self._rps * (window_ms / 1000), window_ms, now_ms]
                )
                if result == 1:
                    return True, "ok"
                return False, f"rate_limit_exceeded ({self._rps} rps)"
            except Exception:
                pass  # Fall through to in-memory

        # In-memory fallback
        if device_id not in _buckets:
            _buckets[device_id] = _TokenBucket(self._rps, self._burst)
        allowed = _buckets[device_id].consume()
        return (True, "ok") if allowed else (False, "rate_limit_exceeded (local)")
