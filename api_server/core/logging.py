"""
SignVerse Unified Logging — Production JSON Schema
Optimized for ELK/Datadog ingestion.
"""

import logging
import json
import time
import os

class JsonFormatter(logging.Formatter):
    def format(self, record):
        log_data = {
            "timestamp": self.formatTime(record),
            "level": record.levelname,
            "message": record.getMessage(),
            "module": record.module,
            "func": record.funcName,
        }
        if hasattr(record, "extra"):
            log_data.update(record.extra)
        return json.dumps(log_data)

def get_logger(name="signverse"):
    logger = logging.getLogger(name)
    logger.setLevel(logging.INFO)
    
    if not logger.handlers:
        handler = logging.StreamHandler()
        handler.setFormatter(JsonFormatter())
        logger.addHandler(handler)
        
    return logger

# Middleware Logger for FastAPI
class APILogger:
    def __init__(self):
        self.logger = get_logger("signverse-api")

    def log_request(self, request_id, method, path, status_code, duration_ms):
        self.logger.info("api_request", extra={
            "request_id": request_id,
            "method": method,
            "path": path,
            "status_code": status_code,
            "duration_ms": duration_ms
        })
