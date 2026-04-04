"""
Pydantic models for type-safe configuration access.
These models validate configuration at runtime and provide IDE autocompletion.
"""
from pydantic import Field, validator, AnyUrl
from pydantic_settings import BaseSettings
from typing import List, Optional, Dict, Any
from pathlib import Path

class AppSettings(BaseSettings):
    """Application settings."""
    name: str = Field("SignVerse", env="APP_NAME")
    version: str = Field("1.0.0", env="APP_VERSION")
    environment: str = Field("development", env="ENVIRONMENT")
    debug: bool = Field(False, env="DEBUG")
    
    api_host: str = Field("0.0.0.0", env="API_HOST")
    api_port: int = Field(8000, env="API_PORT")
    api_workers: int = Field(2, env="API_WORKERS")
    
    cors_origins: List[str] = Field(["http://localhost:3000"], env="CORS_ORIGINS")
    
    class Config:
        env_file = ".env"
        case_sensitive = False
        extra = "ignore"

class LoggingSettings(BaseSettings):
    """Logging configuration."""
    level: str = Field("INFO", env="LOG_LEVEL")
    serialization: str = Field("json", env="LOG_SERIALIZATION")
    
    console_enabled: bool = Field(True, env="LOG_CONSOLE_ENABLED")
    console_level: str = Field("DEBUG", env="LOG_CONSOLE_LEVEL")
    
    file_enabled: bool = Field(False, env="LOG_FILE_ENABLED")
    file_path: Path = Field(Path("./logs/app.log"), env="LOG_FILE_PATH")
    file_level: str = Field("INFO", env="LOG_FILE_LEVEL")
    file_rotation: str = Field("10 MB", env="LOG_FILE_ROTATION")
    
    class Config:
        env_file = ".env"
        extra = "ignore"

class RedisSettings(BaseSettings):
    """Redis connection settings."""
    host: str = Field("localhost", env="REDIS_HOST")
    port: int = Field(6379, env="REDIS_PORT")
    db: int = Field(0, env="REDIS_DB")
    decode_responses: bool = Field(True, env="REDIS_DECODE_RESPONSES")
    
    @property
    def dsn(self) -> str:
        """Return Redis connection string."""
        return f"redis://{self.host}:{self.port}/{self.db}"
    
    class Config:
        env_file = ".env"
        extra = "ignore"

class PoseModelSettings(BaseSettings):
    """Pose estimation model parameters."""
    model_name: str = Field("mediapipe", env="POSE_MODEL_NAME")
    static_image_mode: bool = Field(False, env="POSE_STATIC_IMAGE_MODE")
    model_complexity: int = Field(2, env="POSE_MODEL_COMPLEXITY")
    min_detection_confidence: float = Field(0.5, env="POSE_MIN_DETECTION_CONFIDENCE")
    min_tracking_confidence: float = Field(0.5, env="POSE_MIN_TRACKING_CONFIDENCE")
    
    # Thresholds for different keypoints
    thresholds: Dict[str, float] = Field({
        "nose": 0.5,
        "shoulders": 0.3,
        "elbows": 0.3,
        "wrists": 0.3,
        "hips": 0.3,
        "knees": 0.3,
        "ankles": 0.3
    })
    
    class Config:
        env_file = ".env"
        extra = "ignore"

class PipelineSettings(BaseSettings):
    """Pipeline configuration."""
    max_retries: int = Field(3, env="PIPELINE_MAX_RETRIES")
    timeout_seconds: int = Field(3600, env="PIPELINE_TIMEOUT_SECONDS")
    output_format: str = Field("json", env="PIPELINE_OUTPUT_FORMAT")
    
    interpolation_enabled: bool = Field(True, env="INTERPOLATION_ENABLED")
    interpolation_method: str = Field("linear", env="INTERPOLATION_METHOD")
    
    smoothing_enabled: bool = Field(True, env="SMOOTHING_ENABLED")
    smoothing_window_size: int = Field(5, env="SMOOTHING_WINDOW_SIZE")
    smoothing_polynomial_order: int = Field(2, env="SMOOTHING_POLYNOMIAL_ORDER")
    
    class Config:
        env_file = ".env"
        extra = "ignore"

class YouTubeSettings(BaseSettings):
    """YouTube processing configuration."""
    base_path: Path = Field(Path("./data/youtube"), env="YOUTUBE_BASE_PATH")
    max_concurrent_jobs: int = Field(2, env="YOUTUBE_MAX_CONCURRENT_JOBS")
    cookies_path: Optional[Path] = Field(Path("./configs/cookies.txt"), env="YOUTUBE_COOKIES_PATH")
    
    class Config:
        env_file = ".env"
        extra = "ignore"

# Composite settings model
class Settings(BaseSettings):
    """Main settings container."""
    app: AppSettings = Field(default_factory=AppSettings)
    logging: LoggingSettings = Field(default_factory=LoggingSettings)
    redis: RedisSettings = Field(default_factory=RedisSettings)
    pose_model: PoseModelSettings = Field(default_factory=PoseModelSettings)
    pipeline: PipelineSettings = Field(default_factory=PipelineSettings)
    youtube: YouTubeSettings = Field(default_factory=YouTubeSettings)
    
    @property
    def DATA_DIR(self) -> Path:
        """Alias for root data directory."""
        return Path("./data")
    
    class Config:
        env_file = ".env"
        extra = "ignore"

# Global settings instance
settings = Settings()

# Example usage:
# from core.settings import settings
# print(settings.redis.dsn)
# print(settings.pose_model.min_detection_confidence)
