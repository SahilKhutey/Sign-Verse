"""
Configuration management module for SignVerse System.
Uses Dynaconf for multi-format support and environment-specific configuration.
"""
from pathlib import Path
from dynaconf import Dynaconf, Validator

# Base path for configuration files
config_path = Path(__file__).parent

# Initialize Dynaconf with validation
settings = Dynaconf(
    environments=False,
    envvar_prefix="SIGNVERSE",
    settings_files=[
        str(config_path / "app.yaml"),
        str(config_path / "logging.yaml"), 
        str(config_path / "db.yaml"),
        str(config_path / "model.yaml"),
        str(config_path / "pipeline.yaml"),
        str(config_path / "data.yaml"),
        str(config_path / "simulation.yaml"),
        str(config_path / "api.yaml"),
        str(config_path / "keypoint_labels.yaml"),
    ],
    # Load environment-specific config last (so it overrides)
    includes=[str(config_path / "environment" / f"{env}.yaml") for env in ["default", "dev", "prod"]],
    
    # Validation rules
    validators=[
        Validator("app.environment", is_in=["development", "testing", "production"]),
        Validator("app.version", is_type_of=str),
        Validator("redis.host", is_type_of=str),
        Validator("redis.port", is_type_of=int),
        Validator("pose_estimation.min_detection_confidence", gte=0, lte=1),
        Validator("logging.level", is_in=["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"]),
    ]
)

# Utility function to get configuration
def get_config():
    """Get the current configuration object."""
    return settings

# Example usage: 
# from configs import get_config
# config = get_config()
# redis_host = config.redis.host
