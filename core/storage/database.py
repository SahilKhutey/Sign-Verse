"""
Database storage backend for the SignVerse Storage System.
Handles metadata indexing for videos, frames, persons, and trajectories using SQLAlchemy.
"""
from typing import Dict, List, Any, Optional, Union
from datetime import datetime
from sqlalchemy import create_engine, Column, Integer, Float, String, JSON, DateTime, ForeignKey
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import sessionmaker, relationship
from loguru import logger

from .base import BaseStorageBackend, InitializationError
from ..data_models.skeleton import SkeletonFrame
from ..data_models.trajectory import PersonTrajectory
from ..data_models.storage_schema import PoseStorageSchema, TrajectoryStorageSchema

Base = declarative_base()

class VideoRecord(Base):
    """Metadata for uploaded or downloaded videos."""
    __tablename__ = "videos"
    id = Column(String, primary_key=True)
    filename = Column(String, nullable=False)
    source = Column(String)
    duration = Column(Float)
    fps = Column(Float)
    resolution = Column(String)
    created_at = Column(DateTime, default=datetime.utcnow)
    
    frames = relationship("FrameRecord", back_populates="video")

class FrameRecord(Base):
    """Metadata and pose summary for a single frame."""
    __tablename__ = "frames"
    id = Column(Integer, primary_key=True, autoincrement=True)
    video_id = Column(String, ForeignKey("videos.id"))
    frame_id = Column(Integer, index=True)
    timestamp = Column(Float, index=True)
    person_id = Column(String, index=True)
    
    # Summary metrics
    overall_confidence = Column(Float)
    tracking_quality = Column(Float)
    
    # Store complete pose as JSON for indexing
    pose_data = Column(JSON)
    head_pose = Column(JSON)
    
    video = relationship("VideoRecord", back_populates="frames")

class TrajectoryRecord(Base):
    """Metadata for a complete person trajectory."""
    __tablename__ = "trajectories"
    id = Column(String, primary_key=True)
    person_id = Column(String, index=True)
    video_id = Column(String, index=True)
    start_frame = Column(Integer)
    end_frame = Column(Integer)
    duration = Column(Float)
    frame_count = Column(Integer)
    
    # Summary statistics
    avg_confidence = Column(Float)
    avg_velocity = Column(JSON)
    
    created_at = Column(DateTime, default=datetime.utcnow)

class DatabaseBackend(BaseStorageBackend):
    """Handles metadata indexing in a structured SQL database."""
    
    def __init__(self, config: Dict[str, Any]):
        super().__init__(config)
        self.db_url = config.get("database_url", "sqlite:///./data/signverse.db")
        self.engine = None
        self.Session = None
        
    def initialize(self):
        """Create the database engine and tables."""
        try:
            self.engine = create_engine(self.db_url)
            Base.metadata.create_all(self.engine)
            self.Session = sessionmaker(bind=self.engine)
            self.initialized = True
            logger.info(f"DatabaseBackend initialized at {self.db_url}")
        except Exception as e:
            raise InitializationError(f"Failed to initialize database: {e}")
            
    def store(self, data: Union[SkeletonFrame, PersonTrajectory, VideoRecord], **kwargs) -> bool:
        """Store a record in the database."""
        if not self.initialized:
            self.initialize()
            
        session = self.Session()
        try:
            if isinstance(data, SkeletonFrame):
                record = FrameRecord(
                    video_id=data.source_video or "unknown",
                    frame_id=data.frame_id,
                    timestamp=data.timestamp,
                    person_id=data.person_id,
                    overall_confidence=data.overall_confidence,
                    tracking_quality=data.tracking_quality,
                    pose_data=data.to_dict(),
                    head_pose=data.head_pose.dict() if data.head_pose else None
                )
            elif isinstance(data, PersonTrajectory):
                # Calculate summary stats before storing
                confidences = [p.confidence for p in data.trajectory]
                record = TrajectoryRecord(
                    id=f"{data.person_id}_{data.start_frame}_{data.end_frame}",
                    person_id=data.person_id,
                    video_id=data.metadata.get("source_video", "unknown"),
                    start_frame=data.start_frame,
                    end_frame=data.end_frame,
                    duration=data.trajectory[-1].timestamp - data.trajectory[0].timestamp if data.trajectory else 0,
                    frame_count=len(data.trajectory),
                    avg_confidence=sum(confidences)/len(confidences) if confidences else 0
                )
            elif isinstance(data, Base):
                record = data
            else:
                logger.warning(f"Unsupported data type for DatabaseBackend: {type(data)}")
                return False
                
            session.add(record)
            session.commit()
            return True
        except Exception as e:
            session.rollback()
            logger.error(f"Failed to store database record: {e}")
            return False
        finally:
            session.close()
            
    def retrieve(self, model: Any, filters: Dict[str, Any] = None, limit: int = 100) -> List[Any]:
        """Query the database for records."""
        if not self.initialized:
            self.initialize()
            
        session = self.Session()
        try:
            query = session.query(model)
            if filters:
                for key, value in filters.items():
                    query = query.filter(getattr(model, key) == value)
            return query.limit(limit).all()
        except Exception as e:
            logger.error(f"Failed to query database: {e}")
            return []
        finally:
            session.close()

    def delete(self, model: Any, filters: Dict[str, Any]) -> bool:
        """Delete records from the database."""
        if not self.initialized:
            self.initialize()
            
        session = self.Session()
        try:
            query = session.query(model)
            for key, value in filters.items():
                query = query.filter(getattr(model, key) == value)
            query.delete()
            session.commit()
            return True
        except Exception as e:
            session.rollback()
            logger.error(f"Failed to delete database record: {e}")
            return False
        finally:
            session.close()
