import redis
import json
import os
from config import REDIS_HOST, REDIS_PORT, REDIS_DB

class CacheManager:
    """
    CacheManager — Handles high-performance caching via Redis.
    Used for session state, frequent translation results, and rate-limiting counters.
    """

    def __init__(self):
        try:
            self.client = redis.Redis(
                host=REDIS_HOST,
                port=REDIS_PORT,
                db=REDIS_DB,
                decode_responses=True
            )
            # Test connection
            self.client.ping()
        except redis.ConnectionError:
            print(f"!!! Redis Connection Failed: {REDIS_HOST}:{REDIS_PORT}. Caching disabled.")
            self.client = None

    def get(self, key: str):
        if not self.client: return None
        value = self.client.get(key)
        return json.loads(value) if value else None

    def set(self, key: str, value, expire=300):
        if not self.client: return False
        self.client.set(key, json.dumps(value), ex=expire)
        return True

    def delete(self, key: str):
        if self.client:
            self.client.delete(key)

# Singleton instance
cache_manager = CacheManager()
