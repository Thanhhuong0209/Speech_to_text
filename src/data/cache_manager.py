"""
Cache management utilities for storing and retrieving frequently accessed data.
"""

import os
import json
import pickle
from pathlib import Path
from typing import Any, Dict, Optional, Union
import logging
from datetime import datetime, timedelta

logger = logging.getLogger(__name__)


class CacheManager:
    """Manages caching of various data types for improved performance."""
    
    def __init__(self, cache_dir: str = "cache", max_size_mb: int = 100):
        """
        Initialize the cache manager.
        
        Args:
            cache_dir: Directory to store cache files
            max_size_mb: Maximum cache size in megabytes
        """
        self.cache_dir = Path(cache_dir)
        self.max_size_mb = max_size_mb
        self.cache_dir.mkdir(exist_ok=True)
        
        # Create subdirectories for different cache types
        (self.cache_dir / "audio").mkdir(exist_ok=True)
        (self.cache_dir / "metadata").mkdir(exist_ok=True)
        (self.cache_dir / "transcriptions").mkdir(exist_ok=True)
        
        logger.info(f"Cache manager initialized with directory: {self.cache_dir}")
        
    def get_cache_path(self, cache_type: str, key: str) -> Path:
        """
        Get the full path for a cache file.
        
        Args:
            cache_type: Type of cache (audio, metadata, transcriptions)
            key: Cache key
            
        Returns:
            Full path to cache file
        """
        return self.cache_dir / cache_type / f"{key}.cache"
        
    def set(self, cache_type: str, key: str, value: Any, ttl_hours: int = 24) -> bool:
        """
        Store a value in cache.
        
        Args:
            cache_type: Type of cache
            key: Cache key
            value: Value to cache
            ttl_hours: Time to live in hours
            
        Returns:
            True if successful, False otherwise
        """
        try:
            cache_data = {
                "value": value,
                "timestamp": datetime.now().isoformat(),
                "ttl_hours": ttl_hours
            }
            
            cache_path = self.get_cache_path(cache_type, key)
            
            # Use pickle for complex objects, JSON for simple ones
            if isinstance(value, (dict, list, str, int, float, bool)):
                with open(cache_path, 'w', encoding='utf-8') as f:
                    json.dump(cache_data, f, ensure_ascii=False, indent=2)
            else:
                with open(cache_path, 'wb') as f:
                    pickle.dump(cache_data, f)
                    
            logger.debug(f"Cached {cache_type}:{key}")
            return True
            
        except Exception as e:
            logger.error(f"Error setting cache {cache_type}:{key}: {e}")
            return False
            
    def get(self, cache_type: str, key: str) -> Optional[Any]:
        """
        Retrieve a value from cache.
        
        Args:
            cache_type: Type of cache
            key: Cache key
            
        Returns:
            Cached value if valid, None otherwise
        """
        try:
            cache_path = self.get_cache_path(cache_type, key)
            
            if not cache_path.exists():
                return None
                
            # Try JSON first, then pickle
            try:
                with open(cache_path, 'r', encoding='utf-8') as f:
                    cache_data = json.load(f)
            except (json.JSONDecodeError, UnicodeDecodeError):
                with open(cache_path, 'rb') as f:
                    cache_data = pickle.load(f)
                    
            # Check TTL
            timestamp = datetime.fromisoformat(cache_data["timestamp"])
            ttl = timedelta(hours=cache_data["ttl_hours"])
            
            if datetime.now() - timestamp > ttl:
                logger.debug(f"Cache expired for {cache_type}:{key}")
                self.delete(cache_type, key)
                return None
                
            logger.debug(f"Cache hit for {cache_type}:{key}")
            return cache_data["value"]
            
        except Exception as e:
            logger.error(f"Error getting cache {cache_type}:{key}: {e}")
            return None
            
    def delete(self, cache_type: str, key: str) -> bool:
        """
        Delete a cache entry.
        
        Args:
            cache_type: Type of cache
            key: Cache key
            
        Returns:
            True if successful, False otherwise
        """
        try:
            cache_path = self.get_cache_path(cache_type, key)
            if cache_path.exists():
                cache_path.unlink()
                logger.debug(f"Deleted cache {cache_type}:{key}")
            return True
            
        except Exception as e:
            logger.error(f"Error deleting cache {cache_type}:{key}: {e}")
            return False
            
    def clear_cache_type(self, cache_type: str) -> bool:
        """
        Clear all entries of a specific cache type.
        
        Args:
            cache_type: Type of cache to clear
            
        Returns:
            True if successful, False otherwise
        """
        try:
            cache_type_dir = self.cache_dir / cache_type
            if cache_type_dir.exists():
                for cache_file in cache_type_dir.glob("*.cache"):
                    cache_file.unlink()
                logger.info(f"Cleared {cache_type} cache")
            return True
            
        except Exception as e:
            logger.error(f"Error clearing {cache_type} cache: {e}")
            return False
            
    def clear_all(self) -> bool:
        """
        Clear all cache entries.
        
        Returns:
            True if successful, False otherwise
        """
        try:
            for cache_type_dir in self.cache_dir.iterdir():
                if cache_type_dir.is_dir():
                    for cache_file in cache_type_dir.glob("*.cache"):
                        cache_file.unlink()
            logger.info("Cleared all caches")
            return True
            
        except Exception as e:
            logger.error(f"Error clearing all caches: {e}")
            return False
            
    def get_cache_stats(self) -> Dict[str, Any]:
        """
        Get cache statistics.
        
        Returns:
            Dictionary with cache statistics
        """
        stats = {
            "total_size_mb": 0,
            "cache_types": {},
            "total_files": 0
        }
        
        try:
            for cache_type_dir in self.cache_dir.iterdir():
                if cache_type_dir.is_dir():
                    cache_type = cache_type_dir.name
                    cache_files = list(cache_type_dir.glob("*.cache"))
                    cache_size = sum(f.stat().st_size for f in cache_files)
                    
                    stats["cache_types"][cache_type] = {
                        "files": len(cache_files),
                        "size_mb": round(cache_size / (1024 * 1024), 2)
                    }
                    stats["total_size_mb"] += cache_size / (1024 * 1024)
                    stats["total_files"] += len(cache_files)
                    
            stats["total_size_mb"] = round(stats["total_size_mb"], 2)
            
        except Exception as e:
            logger.error(f"Error getting cache stats: {e}")
            
        return stats
