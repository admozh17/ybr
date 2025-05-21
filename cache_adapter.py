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
import sqlite3


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
# Add to cache_adapter.py
def check_cache_health():
    """Run diagnostics on the cache system and report issues."""
    cache = get_cache_system()
    
    print("------ Cache Health Check ------")
    
    # Check cache system type
    print(f"Cache system type: {type(cache).__name__}")
    
    # Check database file
    db_path = getattr(cache, 'db_path', CACHE_DB_PATH)
    print(f"Cache DB path: {db_path}")
    print(f"Cache DB exists: {os.path.exists(db_path)}")
    
    # Check if directory exists
    db_dir = os.path.dirname(db_path)
    print(f"Cache directory: {db_dir}")
    print(f"Cache directory exists: {os.path.exists(db_dir)}")
    
    # Check DB is writable
    try:
        conn = sqlite3.connect(db_path)
        cursor = conn.cursor()
        cursor.execute("PRAGMA integrity_check")
        integrity = cursor.fetchone()
        print(f"DB integrity check: {integrity[0]}")
        
        # Check tables
        cursor.execute("SELECT name FROM sqlite_master WHERE type='table'")
        tables = cursor.fetchall()
        print(f"Database tables: {[t[0] for t in tables]}")
        
        # Check row counts
        for table in [t[0] for t in tables]:
            cursor.execute(f"SELECT COUNT(*) FROM {table}")
            count = cursor.fetchone()[0]
            print(f"Table {table} has {count} rows")
            
        conn.close()
    except Exception as e:
        print(f"Error checking database: {e}")
    
    # Test cache write/read
    try:
        test_key = "test_health_check"
        test_value = {"test": True, "timestamp": datetime.utcnow().isoformat()}
        test_url = "https://example.com/test-video"
        
        # Test with explicit methods
        if hasattr(cache, 'cache_video_result') and hasattr(cache, 'get_cached_video_result'):
            print("Testing cache_video_result and get_cached_video_result methods...")
            cache.cache_video_result(test_url, test_value)
            result = cache.get_cached_video_result(test_url)
            if result:
                print("✅ Cache method test successful!")
            else:
                print("❌ Cache method test failed - could not read back value")
        
        # Also test dict-like access for fallback implementation
        if hasattr(cache, '__setitem__') and hasattr(cache, '__getitem__'):
            print("Testing dictionary-like access...")
            cache[test_key] = test_value
            try:
                read_value = cache[test_key]
                if read_value:
                    print("✅ Dict-like access test successful!")
                else:
                    print("❌ Dict-like access test failed - returned None")
            except Exception as read_e:
                print(f"❌ Dict-like access test error: {read_e}")
    except Exception as e:
        print(f"Error during cache write/read test: {e}")
    
    print("-------------------------------")
    
    return True

# In cache_adapter.py
def get_cache_system() -> VideoCachingSystem:
    """Get or create the cache system singleton."""
    global _cache_system
    if _cache_system is None:
        # Make sure cache directory exists with absolute path
        cache_dir = os.path.dirname(os.path.abspath(CACHE_DB_PATH))
        if cache_dir:
            os.makedirs(cache_dir, exist_ok=True)
            print(f"Ensuring cache directory exists: {cache_dir}")
        
        try:
            # Try to initialize with the real implementation
            _cache_system = VideoCachingSystem(
                db_path=CACHE_DB_PATH,
                cache_lifetime_days=CACHE_LIFETIME_DAYS
            )
            print(f"Successfully initialized cache system with DB at: {CACHE_DB_PATH}")
            
            # Test the cache with a simple write/read
            test_key = "test_cache_key"
            test_value = {"test": "value", "timestamp": datetime.utcnow().isoformat()}
            
            # Try to write
            if hasattr(_cache_system, '__setitem__'):
                _cache_system[test_key] = test_value
                # Try to read it back
                test_read = _cache_system[test_key] if test_key in _cache_system else None
                if test_read:
                    print(f"Cache test successful! Read back test value.")
                else:
                    print(f"Cache write succeeded but read failed!")
            else:
                print(f"Cache system doesn't support dict-like access")
                
        except Exception as e:
            print(f"ERROR initializing cache system: {str(e)}")
            print(f"Using fallback in-memory cache")
            _cache_system = VideoCachingSystem()  # Fallback
    
    return _cache_system

# In cache_adapter.py
# In cache_adapter.py
def cached_process_video(url, process_func, enable_visual=False):
    """Process a video with caching."""
    # Normalize the URL for consistent cache hits
    cache = get_cache_system()
    
    # First try to normalize the URL for better cache hits
    normalized_url = url
    if hasattr(cache, '_normalize_url'):
        try:
            normalized_url = cache._normalize_url(url)
            print(f"Normalized URL from {url} to {normalized_url}")
        except Exception as e:
            print(f"URL normalization failed: {e}, using original")
    
    # Generate cache key consistently
    url_key = normalized_url.lower().strip()
    cache_key = hashlib.md5(f"{url_key}:{enable_visual}".encode()).hexdigest()
    print(f"Using cache key: {cache_key} for URL: {normalized_url}")
    
    # Try preferred cache retrieval method
    cache_entry = None
    
    # Try the most direct method first - get_cached_video_result
    if hasattr(cache, 'get_cached_video_result'):
        try:
            cache_entry = {"result": cache.get_cached_video_result(normalized_url)}
            if cache_entry["result"]:
                print(f"✅ Retrieved from cache using get_cached_video_result for URL: {normalized_url}")
            else:
                cache_entry = None
        except Exception as e:
            print(f"Error using get_cached_video_result: {e}")
            cache_entry = None
    
    # If the preferred method failed, try alternatives
    if cache_entry is None:
        # Try dictionary access
        try:
            if hasattr(cache, '__getitem__'):
                cache_entry = cache[cache_key]
                print(f"✅ Retrieved from cache using __getitem__ for key: {cache_key}")
        except (KeyError, TypeError, Exception) as e:
            print(f"Cache miss via __getitem__: {str(e)}")
            cache_entry = None
    
    if cache_entry and cache_entry.get("result"):
        # Found in cache - add metadata and return
        result = cache_entry["result"]
        if "_cache" not in result:
            result["_cache"] = {}
        
        result["_cache"].update({
            "cache_hit": True,
            "cache_date": cache_entry.get("timestamp", datetime.utcnow().isoformat()),
            "cache_key": cache_key,
            "retrieval_method": "direct" if hasattr(cache, 'get_cached_video_result') else "dict"
        })
        
        print(f"🎯 Cache hit for URL: {normalized_url}")
        return result
    
    print(f"❌ Cache miss for URL: {normalized_url}")
    
    # Process the video
    result = process_func(normalized_url, enable_visual_recognition=enable_visual)
    
    # Store in cache - try preferred method first (cache_video_result)
    try:
        if hasattr(cache, 'cache_video_result'):
            success = cache.cache_video_result(normalized_url, result)
            if success:
                print(f"✅ Stored in cache using cache_video_result for URL: {normalized_url}")
            else:
                print(f"⚠️ cache_video_result returned False")
        else:
            # Fallback to dictionary-style storage
            timestamp = datetime.utcnow().isoformat()
            cache_entry = {
                "url": normalized_url,
                "result": result,
                "enable_visual": enable_visual,
                "timestamp": timestamp
            }
            
            if hasattr(cache, '__setitem__'):
                cache[cache_key] = cache_entry
                print(f"✅ Stored in cache using __setitem__ for key: {cache_key}")
    except Exception as e:
        print(f"⚠️ Error storing in cache: {str(e)}")
    
    # Add cache metadata to result
    if "_cache" not in result:
        result["_cache"] = {}
    
    result["_cache"].update({
        "cache_hit": False,
        "cache_date": datetime.utcnow().isoformat(),
        "cache_key": cache_key
    })
    
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