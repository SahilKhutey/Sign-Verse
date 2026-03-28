import logging
import json
import time
import os
import sys

class JSONFormatter(logging.Formatter):
    """
    Standardized JSON Formatter for SignVerse microservices.
    Enables easy integration with ELK, Datadog, or Grafana Loki.
    """
    def format(self, record):
        log_data = {
            "timestamp": self.formatTime(record, self.datefmt),
            "level": record.levelname,
            "message": record.getMessage(),
            "logger": record.name,
            "module": record.module,
            "service": os.environ.get("SERVICE_NAME", "signverse-core")
        }
        # Attach structured fields when present
        for key in ("request_id", "path", "method", "status_code", "duration_ms", "client_ip"):
            if hasattr(record, key):
                log_data[key] = getattr(record, key)
        if record.exc_info:
            log_data["exception"] = self.formatException(record.exc_info)
        return json.dumps(log_data)

def get_logger(name: str):
    logger = logging.getLogger(name)
    if not logger.handlers:
        handler = logging.StreamHandler(sys.stdout)
        handler.setFormatter(JSONFormatter())
        logger.addHandler(handler)
        logger.setLevel(logging.INFO)
    return logger
