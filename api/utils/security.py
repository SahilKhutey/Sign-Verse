"""
Security utilities for API authentication and rate limiting.
"""
from fastapi import HTTPException, status, Request
from fastapi.security import APIKeyHeader
from typing import Optional, List
from loguru import logger
import time

from configs import get_config

# Header-based API Key security scheme
api_key_header = APIKeyHeader(name="X-API-Key", auto_error=False)

class RateLimiter:
    """Simple in-memory rate limiting implementation for developer-grade deployments."""
    
    def __init__(self):
        self.requests = {}
        # Load constraints from systems global configs (api.yaml)
        config = get_config().api.rate_limiting
        self.enabled = getattr(config, 'enabled', True)
        self.max_requests = getattr(config, 'max_requests', 100)
        self.time_window = getattr(config, 'time_window', 60)
    
    def check_rate_limit(self, client_id: str) -> bool:
        """Verify if a specific client has exceeded their request quota."""
        if not self.enabled:
            return True
        
        current_time = time.time()
        
        if client_id not in self.requests:
            self.requests[client_id] = []
        
        # Prune expired request timestamps beyond the window
        self.requests[client_id] = [
            req_time for req_time in self.requests[client_id]
            if current_time - req_time < self.time_window
        ]
        
        # Check against system-wide caps
        if len(self.requests[client_id]) >= self.max_requests:
            logger.warning(f"Rate limit exceeded for client: {client_id}")
            return False
        
        # Register the current request
        self.requests[client_id].append(current_time)
        return True

async def validate_api_key(request: Request, api_key: Optional[str] = None):
    """Dependency for validating X-API-Key header against the systems registry."""
    config = get_config().api.security
    
    if not getattr(config, 'enabled', True):
        return True
    
    if not api_key:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Unauthorized: SignVerse API key required in 'X-API-Key' header."
        )
    
    # Verify against allowed keys in api.yaml (and environment overrides)
    allowed_keys = getattr(config, 'allowed_api_keys', [])
    if api_key not in allowed_keys:
        logger.error(f"Invalid API key attempt from {request.client.host}")
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Forbidden: Invalid or expired SignVerse API key."
        )
    
    return True

# Singleton rate limiter for the application cycle
rate_limiter = RateLimiter()
