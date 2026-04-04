"""
Metadata database for fast querying and indexing of SignVerse perception data.
Uses SQLAlchemy to manage Video, Frame, Person, and Trajectory metadata.
"""
from typing import Dict, List, Optional, Any
from datetime import datetime
import numpy as np
from sqlalchemy import create_engine, Column, Integer, Float, String, DateTime, JSON, ForeignKey
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import sessionmaker, relationship
from sqlalchemy.dialects.postgresql import ARRAY
from loguru import logger

Base = declarative_base()

class VideoMetadata(Base):
    """Video metadata table."""
    __tablename__ = "videos"
    
    id = Column(Integer, primary_key=True)
    name = Column(String, unique=True, nullable=False)
    path = Column(String, nullable=False)
    duration = Column(Float)  # seconds
    fps = Column(Float)
    resolution = Column(String)  # "1920x1080"
    frame_count = Column(Integer)
    size_mb = Column(Float)
    source_type = Column(String)  # "upload", "youtube", "external"
    created_at = Column(DateTime, default=datetime.utcnow)
    updated_at = Column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)
    
    # Relationships
    frames = relationship("FrameMetadata", back_populates="video", cascade="all, delete-orphan")
    persons = relationship("PersonMetadata", back_populates="video", cascade="all, delete-orphan")

class FrameMetadata(Base):
    """Frame metadata table."""
    __tablename__ = "frames"
    
    id = Column(Integer, primary_key=True)
    video_id = Column(Integer, ForeignKey("videos.id"), nullable=False)
    frame_number = Column(Integer, nullable=False)
    timestamp = Column(Float, nullable=False)  # seconds
    pose_file_path = Column(String)  # Path to pose JSON file
    person_count = Column(Integer, default=0)
    processing_time = Column(Float)  # seconds
    created_at = Column(DateTime, default=datetime.utcnow)
    
    # Relationships
    video = relationship("VideoMetadata", back_populates="frames")
    persons = relationship("PersonMetadata", back_populates="frame", cascade="all, delete-orphan")

class PersonMetadata(Base):
    """Person metadata table."""
    __tablename__ = "persons"
    
    id = Column(Integer, primary_key=True)
    video_id = Column(Integer, ForeignKey("videos.id"), nullable=False)
    frame_id = Column(Integer, ForeignKey("frames.id"), nullable=False)
    person_id = Column(String, nullable=False)  # "P1", "P2"
    track_id = Column(Integer)  # Internal tracking ID
    bbox = Column(JSON)  # [x1, y1, x2, y2]
    confidence = Column(Float)
    pose_data = Column(JSON)  # Minimal pose data for quick queries
    created_at = Column(DateTime, default=datetime.utcnow)
    
    # Relationships
    video = relationship("VideoMetadata", back_populates="persons")
    frame = relationship("FrameMetadata", back_populates="persons")

class TrajectoryMetadata(Base):
    """Trajectory metadata table."""
    __tablename__ = "trajectories"
    
    id = Column(Integer, primary_key=True)
    video_id = Column(Integer, ForeignKey("videos.id"), nullable=False)
    person_id = Column(String, nullable=False)
    start_frame = Column(Integer, nullable=False)
    end_frame = Column(Integer, nullable=False)
    duration = Column(Float, nullable=False)  # seconds
    frame_count = Column(Integer, nullable=False)
    avg_confidence = Column(Float)
    min_confidence = Column(Float)
    max_confidence = Column(Float)
    joint_list = Column(JSON)  # Using JSON instead of ARRAY(String) for SQLite compatibility
    trajectory_path = Column(String)  # Path to trajectory data
    created_at = Column(DateTime, default=datetime.utcnow)
    updated_at = Column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)

class MetadataDB:
    """Manages metadata database operations."""
    
    def __init__(self, database_url: str = "sqlite:///./data/metadata.db"):
        """
        Initialize metadata database.
        
        Args:
            database_url: Database connection URL
        """
        self.engine = create_engine(database_url)
        self.Session = sessionmaker(bind=self.engine)
        
        # Create tables
        Base.metadata.create_all(self.engine)
        
        logger.info(f"Initialized metadata database: {database_url}")
    
    def add_video(self, name: str, path: str, duration: float, fps: float, 
                 resolution: str, frame_count: int, size_mb: float, 
                 source_type: str = "upload") -> VideoMetadata:
        """Add video metadata."""
        session = self.Session()
        try:
            video = VideoMetadata(
                name=name,
                path=path,
                duration=duration,
                fps=fps,
                resolution=resolution,
                frame_count=frame_count,
                size_mb=size_mb,
                source_type=source_type
            )
            session.add(video)
            session.commit()
            session.refresh(video)
            return video
        except Exception as e:
            session.rollback()
            logger.error(f"Failed to add video metadata: {e}")
            raise
        finally:
            session.close()
    
    def add_frame(self, video_id: int, frame_number: int, timestamp: float,
                 pose_file_path: str, person_count: int = 0, 
                 processing_time: float = 0.0) -> FrameMetadata:
        """Add frame metadata."""
        session = self.Session()
        try:
            frame = FrameMetadata(
                video_id=video_id,
                frame_number=frame_number,
                timestamp=timestamp,
                pose_file_path=pose_file_path,
                person_count=person_count,
                processing_time=processing_time
            )
            session.add(frame)
            session.commit()
            session.refresh(frame)
            return frame
        except Exception as e:
            session.rollback()
            logger.error(f"Failed to add frame metadata: {e}")
            raise
        finally:
            session.close()
    
    def add_person(self, video_id: int, frame_id: int, person_id: str,
                  track_id: int, bbox: List[float], confidence: float,
                  pose_data: Dict[str, Any]) -> PersonMetadata:
        """Add person metadata."""
        session = self.Session()
        try:
            person = PersonMetadata(
                video_id=video_id,
                frame_id=frame_id,
                person_id=person_id,
                track_id=track_id,
                bbox=bbox,
                confidence=confidence,
                pose_data=pose_data
            )
            session.add(person)
            session.commit()
            session.refresh(person)
            return person
        except Exception as e:
            session.rollback()
            logger.error(f"Failed to add person metadata: {e}")
            raise
        finally:
            session.close()
    
    def add_trajectory(self, video_id: int, person_id: str, start_frame: int,
                      end_frame: int, duration: float, frame_count: int,
                      avg_confidence: float, min_confidence: float,
                      max_confidence: float, joint_list: List[str],
                      trajectory_path: str) -> TrajectoryMetadata:
        """Add trajectory metadata."""
        session = self.Session()
        try:
            trajectory = TrajectoryMetadata(
                video_id=video_id,
                person_id=person_id,
                start_frame=start_frame,
                end_frame=end_frame,
                duration=duration,
                frame_count=frame_count,
                avg_confidence=avg_confidence,
                min_confidence=min_confidence,
                max_confidence=max_confidence,
                joint_list=joint_list,
                trajectory_path=trajectory_path
            )
            session.add(trajectory)
            session.commit()
            session.refresh(trajectory)
            return trajectory
        except Exception as e:
            session.rollback()
            logger.error(f"Failed to add trajectory metadata: {e}")
            raise
        finally:
            session.close()
    
    def query_frames_with_person(self, person_id: str, video_name: Optional[str] = None,
                                min_confidence: float = 0.5) -> List[FrameMetadata]:
        """Query frames containing a specific person."""
        session = self.Session()
        try:
            query = session.query(FrameMetadata).join(PersonMetadata).filter(
                PersonMetadata.person_id == person_id,
                PersonMetadata.confidence >= min_confidence
            )
            
            if video_name:
                query = query.join(VideoMetadata).filter(VideoMetadata.name == video_name)
            
            return query.order_by(FrameMetadata.timestamp).all()
        finally:
            session.close()
    
    def query_person_trajectories(self, video_name: str, 
                                 min_duration: float = 1.0) -> List[TrajectoryMetadata]:
        """Query person trajectories for a video."""
        session = self.Session()
        try:
            return session.query(TrajectoryMetadata).join(VideoMetadata).filter(
                VideoMetadata.name == video_name,
                TrajectoryMetadata.duration >= min_duration
            ).order_by(TrajectoryMetadata.person_id).all()
        finally:
            session.close()
    
    def get_video_stats(self, video_name: str) -> Dict[str, Any]:
        """Get statistics for a video."""
        session = self.Session()
        try:
            video = session.query(VideoMetadata).filter_by(name=video_name).first()
            if not video:
                return {}
            
            # Get person count
            person_count = session.query(PersonMetadata.person_id).filter_by(
                video_id=video.id
            ).distinct().count()
            
            # Get trajectory stats
            trajectories = session.query(TrajectoryMetadata).filter_by(video_id=video.id).all()
            
            return {
                "video_id": video.id,
                "name": video.name,
                "person_count": person_count,
                "trajectory_count": len(trajectories),
                "avg_trajectory_duration": np.mean([t.duration for t in trajectories]) if trajectories else 0,
                "avg_confidence": np.mean([t.avg_confidence for t in trajectories]) if trajectories else 0
            }
        finally:
            session.close()
