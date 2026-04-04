"""
Base interfaces and abstract classes for the SignVerse Storage System.
"""
from abc import ABC, abstractmethod
from typing import Dict, List, Any, Optional
from loguru import logger

class BaseStorageBackend(ABC):
    """Abstract base class for all storage backends (File, DB, High-Performance)."""
    
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.initialized = False
        
    @abstractmethod
    def initialize(self):
        """Initialize the storage backend (connect to DB, create directories, etc.)."""
        pass
        
    @abstractmethod
    def store(self, data: Any, **kwargs) -> bool:
        """Store data in the backend."""
        pass
        
    @abstractmethod
    def retrieve(self, query: Any, **kwargs) -> Any:
        """Retrieve data from the backend."""
        pass
        
    @abstractmethod
    def delete(self, identifier: Any, **kwargs) -> bool:
        """Delete data from the backend."""
        pass

class StorageError(Exception):
    """Base exception for storage-related errors."""
    pass

class InitializationError(StorageError):
    """Raised when a storage backend fails to initialize."""
    pass
