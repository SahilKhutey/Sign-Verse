import hashlib
import os

class TranslationCache:
    """Cache frequent translations for faster response"""
    
    def __init__(self, max_size=10000):
        self.cache = {}
        self.lru = []
        self.max_size = max_size
        
        # Use embeddings for similarity search (requires sentence-transformers)
        self.embedding_model = None
        try:
            from sentence_transformers import SentenceTransformer
            self.embedding_model = SentenceTransformer('all-MiniLM-L6-v2')
        except ImportError:
            print("SentenceTransformer not found. Cache will use exact matches.")
    
    def get(self, input_data, input_type="sign"):
        """Get from cache or compute"""
        key = self._generate_key(input_data, input_type)
        
        # Check cache
        if key in self.cache:
            # Update LRU
            self.lru.remove(key)
            self.lru.append(key)
            return self.cache[key]
        
        # Not in cache, compute
        return None
    
    def set(self, input_data, result, input_type="sign"):
        """Store in cache"""
        key = self._generate_key(input_data, input_type)
        
        # Add to cache
        self.cache[key] = result
        self.lru.append(key)
        
        # Evict if needed
        if len(self.cache) > self.max_size:
            oldest = self.lru.pop(0)
            del self.cache[oldest]

    def _generate_key(self, data, input_type):
        if input_type == "text":
            return hashlib.md5(data.encode('utf-8')).hexdigest()
        elif input_type == "sign":
            # For sign data (numpy/tensors), we hash the flattened values
            import numpy as np
            if hasattr(data, 'numpy'):
                data = data.numpy()
            return hashlib.md5(data.tobytes()).hexdigest()
        elif input_type == "audio":
            return hashlib.md5(data).hexdigest()
        return str(hash(data))
