#!/usr/bin/env python3
"""
Adapter module for integrating the caching system with the web application.
"""

import os
import time
import hashlib
import json
from datetime import datetime
from typing import Dict, Any, Optional, List

# Import the actual caching system
try:
    from caching_system import VideoCachingSystem
except ImportError:
    # Fallback simple implementation if the real one isn't available
    class VideoCachingSystem:
        def __init__(self, db_path=None, cache_lifetime_days=30):
            self.cache = {}
            self.db_path = db_path
            self.cache_lifetime_days = cache_lifetime_days
            print(f"⚠️ Using simplified in-memory cache (no persistent storage)")
            
        def __getitem__(self, key):
            return self.cache.get(key)
            
        def __setitem__(self, key, value):
            self.cache[key] = value
            
        def get_cached_place_info(self, place_name, country=None, city=None, street_address=None, region=None):
            # Simple implementation - in real version would do fuzzy matching
            return None
            
        def get_cache_stats(self):
            return {
                "total_entries": len(self.cache),
                "db_size_mb": 0.0,
                "hits": 0,
                "misses": 0,
                "hit_ratio": 0.0,
                "oldest_entry": None,
                "newest_entry": None
            }
            
        def cleanup_old_entries(self, days=None):
            # In-memory version doesn't need cleanup
            pass

# Initialize caching system with configurable path
CACHE_DB_PATH = os.environ.get("CACHE_DB_PATH", "cache/analysis_cache.db")
CACHE_LIFETIME_DAYS = int(os.environ.get("CACHE_LIFETIME_DAYS", "30"))

# Ensure cache directory exists
os.makedirs(os.path.dirname(CACHE_DB_PATH), exist_ok=True)

# Create singleton cache instance
_cache_system = None

def get_cache_system() -> VideoCachingSystem:
    """
    Get or create the cache system singleton.
    
    Returns:
        VideoCachingSystem instance
    """
    global _cache_system
    if _cache_system is None:
        _cache_system = VideoCachingSystem(
            db_path=CACHE_DB_PATH,
            cache_lifetime_days=CACHE_LIFETIME_DAYS
        )
    return _cache_system

def cached_process_video(url, process_func, enable_visual=False):
    """
    Process a video with caching to avoid repeated processing of the same URL.
    
    Args:
        url: The URL to process
        process_func: Function that takes a URL and optional kwargs
        enable_visual: Whether to enable visual recognition
        
    Returns:
        Processing result, either from cache or freshly processed
    """
    # Generate a cache key based on the URL and enable_visual flag
    url_key = url.lower().strip()
    cache_key = hashlib.md5(f"{url_key}:{enable_visual}".encode()).hexdigest()
    
    # Get the cache system
    cache = get_cache_system()
    
    # Try to get cache entry - try multiple possible methods to be compatible
    cache_entry = None
    try:
        # First try the method we expected
        if hasattr(cache, 'get_cache_entry'):
            cache_entry = cache.get_cache_entry(cache_key)
        # Next try dictionary style access
        elif hasattr(cache, '__getitem__'):
            try:
                cache_entry = cache[cache_key]
            except (KeyError, TypeError):
                cache_entry = None
        # Finally try a 'db' attribute if it exists
        elif hasattr(cache, 'db'):
            if hasattr(cache.db, 'get'):
                cache_entry = cache.db.get(cache_key)
            elif hasattr(cache.db, '__getitem__'):
                try:
                    cache_entry = cache.db[cache_key]
                except (KeyError, TypeError):
                    cache_entry = None
    except Exception as e:
        print(f"Warning: Error accessing cache: {e}")
        cache_entry = None
    
    if cache_entry:
        # Found in cache
        result = cache_entry["result"]
        
        # Add cache metadata to the result
        if "_cache" not in result:
            result["_cache"] = {}
        
        result["_cache"].update({
            "cache_hit": True,
            "cache_date": cache_entry["timestamp"],
            "cache_key": cache_key
        })
        
        print(f"🎯 Cache hit for URL: {url}")
        return result
        
    print(f"❌ Cache miss for URL: {url}")
    
    # Process the video
    result = process_func(url, enable_visual_recognition=enable_visual)
    
    # Store the result in the cache
    timestamp = datetime.utcnow().isoformat()
    cache_entry = {
        "url": url,
        "result": result,
        "enable_visual": enable_visual,
        "timestamp": timestamp
    }
    
    # Try different methods to set the cache entry
    try:
        if hasattr(cache, 'set_cache_entry'):
            cache.set_cache_entry(cache_key, cache_entry)
        elif hasattr(cache, '__setitem__'):
            cache[cache_key] = cache_entry
        elif hasattr(cache, 'db'):
            if hasattr(cache.db, '__setitem__'):
                cache.db[cache_key] = cache_entry
            # Also try if there's an update method
            elif hasattr(cache.db, 'update'):
                cache.db.update({cache_key: cache_entry})
    except Exception as e:
        print(f"Warning: Error updating cache: {e}")
    
    # Add cache metadata to the result
    if "_cache" not in result:
        result["_cache"] = {}
    
    result["_cache"].update({
        "cache_hit": False,
        "cache_date": timestamp,
        "cache_key": cache_key
    })
    
    # Return the result
    return result

def get_place_info(place_name: str, country: Optional[str] = None, city: Optional[str] = None, 
                   street_address: Optional[str] = None, region: Optional[str] = None) -> Optional[Dict[str, Any]]:
    """
    Get cached place information for external use.
    
    Args:
        place_name: Name of the place
        country: Optional country for better matching
        city: Optional city for better matching
        street_address: Optional street address for exact location matching
        region: Optional region for additional context
        
    Returns:
        Place information or None if not found
    """
    cache = get_cache_system()
    return cache.get_cached_place_info(place_name, country, city, street_address, region)

def get_cache_stats() -> Dict[str, Any]:
    """
    Get cache statistics for monitoring.
    
    Returns:
        Dictionary with cache statistics
    """
    cache = get_cache_system()
    return cache.get_cache_stats()

def cleanup_cache(days: Optional[int] = None) -> None:
    """
    Clean up old cache entries.
    
    Args:
        days: Age threshold in days (uses default if None)
    """
    cache = get_cache_system()
    cache.cleanup_old_entries(days)