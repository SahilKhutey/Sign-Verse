"""
Database Models — User and TranslationHistory.
"""

from sqlalchemy import Column, Integer, String, DateTime, ForeignKey, Text, Boolean
from sqlalchemy.orm import relationship
from datetime import datetime
import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
from database.db_connection import Base


class User(Base):
    __tablename__ = "users"

    id = Column(Integer, primary_key=True, index=True)
    username = Column(String(50), unique=True, index=True)
    email = Column(String(100), unique=True, index=True)
    hashed_password = Column(String(255))
    created_at = Column(DateTime, default=datetime.utcnow)

    translations = relationship("TranslationHistory", back_populates="user")


class TranslationHistory(Base):
    __tablename__ = "translation_history"

    id = Column(Integer, primary_key=True, index=True)
    user_id = Column(Integer, ForeignKey("users.id"))
    input_text = Column(Text)
    sign_gloss = Column(Text)
    translation_type = Column(String(50))  # speech_to_sign, text_to_sign, sign_to_text
    created_at = Column(DateTime, default=datetime.utcnow)

    user = relationship("User", back_populates="translations")


class RefreshToken(Base):
    __tablename__ = "refresh_tokens"

    id = Column(Integer, primary_key=True, index=True)
    user_id = Column(Integer, ForeignKey("users.id"), index=True)
    token_hash = Column(String(255), unique=True, index=True)
    expires_at = Column(DateTime, nullable=False)
    revoked = Column(Boolean, default=False)
    created_at = Column(DateTime, default=datetime.utcnow)
