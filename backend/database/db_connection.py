"""
Database Connection — SQLite via SQLAlchemy.
"""

from sqlalchemy import create_engine
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import sessionmaker
import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
from config import DATABASE_URL

# Enable connection pooling for production (PostgreSQL)
connect_args = {"check_same_thread": False} if DATABASE_URL.startswith("sqlite") else {}

engine = create_engine(
    DATABASE_URL, 
    connect_args=connect_args,
    pool_size=20,
    max_overflow=10,
    pool_timeout=30
)

SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)

Base = declarative_base()


def init_db():
    """Initialize database tables."""
    from database.models import User, TranslationHistory
    Base.metadata.create_all(bind=engine)


def get_db():
    """Dependency: get database session."""
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()
