#!/usr/bin/env python3
"""
Caching system for video analysis results to avoid redundant processing.
Implements two levels of caching:
1. Video URL-based caching (exact matches)
2. Place-based caching (partial matches from different videos)
"""

import json
import hashlib
import time
import os
import sqlite3
import pathlib
from typing import Dict, Any, List, Optional, Tuple
from datetime import datetime, timedelta

class VideoCachingSystem:
    """
    Caching system for video analysis with two-level architecture:
    - URL-based exact matching: Reuse entire results for same video
    - Place-based partial matching: Reuse place info across different videos
    """
    
    def __init__(self, db_path: str = "cache.db", cache_lifetime_days: int = 30):
        """
        Initialize the caching system.
        
        Args:
            db_path: Path to SQLite database file
            cache_lifetime_days: Number of days before a cache entry expires
        """
        self.db_path = db_path
        self.cache_lifetime_days = cache_lifetime_days
        self._initialize_db()
    
    def _initialize_db(self):
        """Create necessary database tables if they don't exist."""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        # Table for URL-based video caching
        cursor.execute('''
        CREATE TABLE IF NOT EXISTS video_cache (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            url TEXT UNIQUE NOT NULL,
            url_hash TEXT UNIQUE NOT NULL,
            result_json TEXT NOT NULL,
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            last_accessed TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        )
        ''')
        
        # Table for place-based caching
        cursor.execute('''
        CREATE TABLE IF NOT EXISTS place_cache (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            place_name TEXT NOT NULL,
            place_normalized TEXT NOT NULL,
            genre TEXT,
            country TEXT,
            city TEXT,
            result_json TEXT NOT NULL,
            source_video_url TEXT NOT NULL,
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            last_accessed TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        )
        ''')
        
        # Create indexes for faster lookups
        cursor.execute('CREATE INDEX IF NOT EXISTS idx_video_url ON video_cache(url_hash)')
        cursor.execute('CREATE INDEX IF NOT EXISTS idx_place_name ON place_cache(place_normalized)')
        cursor.execute('CREATE INDEX IF NOT EXISTS idx_place_location ON place_cache(country, city)')
        
        # Enable JSON functions for SQLite (for better place matching)
        cursor.execute('PRAGMA journal_mode=WAL')
        
        # Create unique compound index for location-aware place lookup
        try:
            cursor.execute('CREATE UNIQUE INDEX IF NOT EXISTS idx_place_unique_location ON place_cache(place_normalized, country, city, json_extract(result_json, "$.availability.street_address"))')
        except sqlite3.OperationalError:
            # If JSON functions not supported in this SQLite version, use simpler index
            print("Note: Using simplified place index - your SQLite version may not support JSON functions")
            cursor.execute('DROP INDEX IF EXISTS idx_place_unique_location')
            cursor.execute('CREATE UNIQUE INDEX IF NOT EXISTS idx_place_simple ON place_cache(place_normalized, country, city)')
        
        conn.commit()
        conn.close()
    

    
    def _normalize_url(self, url: str) -> str:
        """
        Normalize URL for more consistent caching.
        
        Args:
            url: Original URL
            
        Returns:
            Normalized URL
        """
        import re
        
        # Debug
        print(f"🔍 Normalizing URL: {url}")
        
        # For Instagram Posts/Reels, normalize to a standard format
        instagram_match = re.search(r'instagram\.com/(?:p|reel)/([^/?]+)', url)
        if instagram_match:
            post_id = instagram_match.group(1)
            normalized = f"https://www.instagram.com/p/{post_id}/"
            print(f"📋 Normalized Instagram URL: {normalized}")
            return normalized
        
        # For TikTok, extract just the video ID
        tiktok_match = re.search(r'tiktok\.com/.*?/video/(\d+)', url)
        if tiktok_match:
            video_id = tiktok_match.group(1)
            return f"https://www.tiktok.com/video/{video_id}"
        
        # For YouTube Shorts, standardize format
        youtube_match = re.search(r'youtube\.com/shorts/([^/?&]+)', url) or re.search(r'youtu\.be/([^/?&]+)', url)
        if youtube_match:
            video_id = youtube_match.group(1)
            return f"https://www.youtube.com/shorts/{video_id}"
        
        # For all other URLs, return as is
        return url
    
    def _hash_url(self, url: str) -> str:
        """
        Create a hash of the URL for faster lookups.
        
        Args:
            url: Video URL
            
        Returns:
            SHA-256 hash of the normalized URL
        """
        normalized_url = self._normalize_url(url)
        url_hash = hashlib.sha256(normalized_url.encode()).hexdigest()
        # Add debug print
        print(f"🔑 Generated hash: {url_hash} for normalized URL: {normalized_url}")
        return url_hash
    def _normalize_place_name(self, name: str) -> str:
        """
        Normalize a place name to a lowercase, underscore-separated key.

        Args:
            name: The place name

        Returns:
            Normalized place name string
        """
        return (
            name.strip()
                .lower()
                .replace(" ", "_")
                .replace("-", "_")
                .replace("’", "")
                .replace("'", "")
        )
    
    def get_cached_video_result(self, url: str) -> Optional[Dict[str, Any]]:
        """
        Get cached result for a video URL.
        
        Args:
            url: Video URL
            
        Returns:
            Cached result as a dictionary or None if not found
        """
        url_hash = self._hash_url(url)
        
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        # Check if URL exists in cache
        cursor.execute('''
        SELECT result_json, created_at FROM video_cache 
        WHERE url_hash = ? 
        ''', (url_hash,))
        
        result = cursor.fetchone()
        
        if result:
            result_json, created_at = result
            
            # Check if cache is expired
            cache_date = datetime.strptime(created_at, '%Y-%m-%d %H:%M:%S')
            expiry_date = cache_date + timedelta(days=self.cache_lifetime_days)
            
            if datetime.now() > expiry_date:
                # Cache is expired, remove it
                cursor.execute('DELETE FROM video_cache WHERE url_hash = ?', (url_hash,))
                conn.commit()
                conn.close()
                return None
            
            # Update last accessed timestamp
            cursor.execute('''
            UPDATE video_cache SET last_accessed = CURRENT_TIMESTAMP
            WHERE url_hash = ?
            ''', (url_hash,))
            conn.commit()
            conn.close()
            
            # Return cached result
            return json.loads(result_json)
        
        conn.close()
        return None
    
    def cache_video_result(self, url: str, result: Dict[str, Any]) -> bool:
        """
        Cache result for a video URL and extract places for the place cache.
        
        Args:
            url: Video URL
            result: Analysis result dictionary
            
        Returns:
            True if successfully cached, False otherwise
        """
        normalized_url = self._normalize_url(url)
        url_hash = self._hash_url(url)  # This now uses normalized_url internally
        
        # Add debug print statements
        print(f"Storing with URL hash: {url_hash} for normalized URL: {normalized_url}")
        
        # Update the result with the normalized URL
        if "url" in result:
            result["url"] = normalized_url
        
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        try:
            # Insert or replace in video cache
            cursor.execute('''
            INSERT OR REPLACE INTO video_cache 
            (url, url_hash, result_json, created_at, last_accessed) 
            VALUES (?, ?, ?, CURRENT_TIMESTAMP, CURRENT_TIMESTAMP)
            ''', (normalized_url, url_hash, json.dumps(result)))
            
            # Add debug success message
            print(f"✅ Stored result in cache for hash: {url_hash}")
            
            # Extract places from activities for place cache
            if "activities" in result:
                for activity in result["activities"]:
                    place_name = activity.get("place_name")
                    if place_name:
                        # Normalize place name
                        place_normalized = self._normalize_place_name(place_name)
                        
                        # Get location data
                        availability = activity.get("availability", {})
                        country = availability.get("country", "")
                        city = availability.get("city", "")
                        street_address = availability.get("street_address", "")
                        region = availability.get("region", "")
                        genre = activity.get("genre", "")
                        
                        # Skip if insufficient data
                        if not place_normalized:
                            continue
                        
                        # Check if this place already exists with a different address
                        if street_address:
                            cursor.execute('''
                            SELECT place_name, json_extract(result_json, '$.availability.street_address') as address 
                            FROM place_cache 
                            WHERE place_normalized = ? 
                            AND city = ? 
                            AND country = ?
                            AND json_extract(result_json, '$.availability.street_address') != ''
                            AND json_extract(result_json, '$.availability.street_address') != ?
                            ''', (place_normalized, city, country, street_address))
                            
                            conflicts = cursor.fetchall()
                            if conflicts:
                                # There's a conflict - this is likely a different location with the same name
                                # Create a more specific normalized name incorporating the location
                                location_suffix = f"_{self._normalize_place_name(city or country or region)}"
                                place_normalized = place_normalized + location_suffix
                                
                                print(f"⚠️ Name conflict detected for '{place_name}'")
                                print(f"  - New address: '{street_address}'")
                                print(f"  - Existing entries: {conflicts}")
                                print(f"  - Using location-specific key: '{place_normalized}'")
                        
                        try:
                            # Create unique cache key for disambiguation
                            cache_key = place_normalized
                            if city or country:
                                # Add location info to the key to prevent conflicts
                                location_part = "_".join(filter(None, [
                                    self._normalize_place_name(city) if city else "",
                                    self._normalize_place_name(country) if country else ""
                                ]))
                                if location_part:
                                    cache_key = f"{place_normalized}_{location_part}"
                            
                            # Insert or replace in place cache
                            cursor.execute('''
                            INSERT OR REPLACE INTO place_cache 
                            (place_name, place_normalized, genre, country, city, result_json, source_video_url)
                            VALUES (?, ?, ?, ?, ?, ?, ?)
                            ''', (
                                place_name, 
                                cache_key, 
                                genre, 
                                country, 
                                city, 
                                json.dumps(activity), 
                                normalized_url
                            ))
                        except Exception as e:
                            print(f"Error caching place '{place_name}': {e}")
            
            conn.commit()
            return True
        except Exception as e:
            print(f"Error caching video result: {e}")
            conn.rollback()
            return False
        finally:
            conn.close()
    
    def get_cached_place_info(self, place_name: str, country: Optional[str] = None, city: Optional[str] = None, 
                              street_address: Optional[str] = None, region: Optional[str] = None) -> Optional[Dict[str, Any]]:
        """
        Get cached information for a place with strict location differentiation.
        
        Args:
            place_name: Name of the place to look up
            country: Optional country for more precise matching
            city: Optional city for more precise matching
            street_address: Optional street address for exact location matching
            region: Optional region/area for additional context
            
        Returns:
            Cached place information or None if not found
        """
        if not place_name:
            return None
        
        # Normalize place name
        place_normalized = self._normalize_place_name(place_name)
        
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        # First try: Exact location match (including street address if available)
        if street_address:
            # Look for exact address match first
            cursor.execute('''
            SELECT result_json, place_name FROM place_cache 
            WHERE place_normalized = ? 
            AND (json_extract(result_json, '$.availability.street_address') = ? OR json_extract(result_json, '$.availability.street_address') LIKE ?)
            ''', (place_normalized, street_address, f"%{street_address}%"))
            
            result = cursor.fetchone()
            if result:
                place_info = json.loads(result[0])
                orig_name = result[1]
                print(f"✅ Exact address match found for '{place_name}' at '{street_address}'")
                
                # Update last accessed timestamp
                cursor.execute("UPDATE place_cache SET last_accessed = CURRENT_TIMESTAMP WHERE place_normalized = ? AND json_extract(result_json, '$.availability.street_address') = ?", 
                              (place_normalized, street_address))
                conn.commit()
                conn.close()
                return place_info
        
        # Second try: Location-based match (city + country)
        location_filters = []
        params = [place_normalized]
        
        if city and city.strip():
            location_filters.append("(city = ? OR json_extract(result_json, '$.availability.city') = ?)")
            params.extend([city, city])
        
        if country and country.strip():
            location_filters.append("(country = ? OR json_extract(result_json, '$.availability.country') = ?)")
            params.extend([country, country])
            
        if region and region.strip():
            location_filters.append("(region = ? OR json_extract(result_json, '$.availability.region') = ?)")
            params.extend([region, region])
        
        # Construct query with location filters
        query = 'SELECT result_json, place_name FROM place_cache WHERE place_normalized = ?'
        
        if location_filters:
            query += ' AND ' + ' AND '.join(location_filters)
            
        # Order by exact name match and recency
        query += ' ORDER BY CASE WHEN place_name = ? THEN 0 ELSE 1 END, last_accessed DESC LIMIT 1'
        params.append(place_name)
        
        cursor.execute(query, params)
        result = cursor.fetchone()
        
        if result:
            place_info = json.loads(result[0])
            orig_name = result[1]
            
            # Verify this is really the same place by checking address conflicts
            cached_street = place_info.get("availability", {}).get("street_address", "")
            
            # If both have street addresses and they don't match, likely different locations
            if street_address and cached_street and street_address.lower() not in cached_street.lower() and cached_street.lower() not in street_address.lower():
                print(f"⚠️ Rejected cache match: '{place_name}' at '{street_address}' conflicts with cached address '{cached_street}'")
                conn.close()
                return None
            
            print(f"✅ Location match found for '{place_name}' in {city or ''}, {country or ''}")
            
            # Update last accessed timestamp
            location_update = []
            update_params = [place_normalized]
            
            if city:
                location_update.append("(city = ? OR json_extract(result_json, '$.availability.city') = ?)")
                update_params.extend([city, city])
            
            if country:
                location_update.append("(country = ? OR json_extract(result_json, '$.availability.country') = ?)")
                update_params.extend([country, country])
                
            if region:
                location_update.append("(region = ? OR json_extract(result_json, '$.availability.region') = ?)")
                update_params.extend([region, region])
                
            update_query = "UPDATE place_cache SET last_accessed = CURRENT_TIMESTAMP WHERE place_normalized = ?"
            
            if location_update:
                update_query += ' AND ' + ' AND '.join(location_update)
                
            cursor.execute(update_query, update_params)
            conn.commit()
            conn.close()
            return place_info
        
        # No match found
        print(f"❌ No matching place found for '{place_name}' in {city or ''}, {country or ''}")
        conn.close()
        return None
    
    def merge_with_cached_places(self, result: Dict[str, Any]) -> Dict[str, Any]:
        """
        Merge analysis result with cached place information.
        
        Args:
            result: Fresh analysis result
            
        Returns:
            Merged result with cached place data where appropriate
        """
        if "activities" not in result:
            return result
        
        merged_result = result.copy()
        activities = []
        
        for activity in result["activities"]:
            place_name = activity.get("place_name")
            if not place_name:
                activities.append(activity)
                continue
            
            # Get location data for better matching
            availability = activity.get("availability", {})
            country = availability.get("country", "")
            city = availability.get("city", "")
            
            # Look for cached place info
            cached_place = self.get_cached_place_info(place_name, country, city)
            
            if cached_place:
                # Merge with cached place data
                merged_activity = self._merge_place_data(activity, cached_place)
                activities.append(merged_activity)
                
                print(f"✅ Merged with cached place: {place_name}")
            else:
                # Use original activity data
                activities.append(activity)
        
        merged_result["activities"] = activities
        return merged_result
    
    def _merge_place_data(self, new_data: Dict[str, Any], cached_data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Intelligently merge new activity data with cached place data.
        Prefers complete cached data while preserving fresh visual and unique aspects.
        
        Args:
            new_data: New activity data from current analysis
            cached_data: Cached place data from previous analyses
            
        Returns:
            Merged activity data
        """
        merged = new_data.copy()
        
        # First, verify we're dealing with the same place by checking addresses
        new_address = new_data.get("availability", {}).get("street_address", "").lower()
        cached_address = cached_data.get("availability", {}).get("street_address", "").lower()
        
        # If both have addresses and they're completely different, abort merging
        if new_address and cached_address and new_address not in cached_address and cached_address not in new_address:
            # Check if at least city and country match before declaring conflict
            new_city = new_data.get("availability", {}).get("city", "").lower()
            cached_city = cached_data.get("availability", {}).get("city", "").lower()
            new_country = new_data.get("availability", {}).get("country", "").lower()
            cached_country = cached_data.get("availability", {}).get("country", "").lower()
            
            # If cities or countries conflict, we're dealing with different places with the same name
            if (new_city and cached_city and new_city != cached_city) or \
               (new_country and cached_country and new_country != cached_country):
                print(f"⚠️ Place name conflict detected: Same name but different locations")
                print(f"  - New: {new_address}, {new_city}, {new_country}")
                print(f"  - Cached: {cached_address}, {cached_city}, {cached_country}")
                
                # Log this conflict to help improve the system
                place_name = new_data.get("place_name", "unknown")
                print(f"⚠️ Not merging data for '{place_name}' - appears to be different locations")
                
                # Return the new data unmodified
                return new_data
        
        # Always prefer cached genre if available and new data doesn't have a specific one
        if cached_data.get("genre") and (not new_data.get("genre") or new_data.get("genre") == "other"):
            merged["genre"] = cached_data["genre"]
        
        # Always prefer cached cuisine if available
        if cached_data.get("cuisine") and not new_data.get("cuisine"):
            merged["cuisine"] = cached_data["cuisine"]
        
        # Merge availability info (prefer cached detailed address)
        if "availability" in cached_data and "availability" in new_data:
            cached_avail = cached_data["availability"]
            new_avail = new_data["availability"]
            
            merged_avail = new_avail.copy()
            
            # Prefer cached street address if available
            if cached_avail.get("street_address") and not new_avail.get("street_address"):
                merged_avail["street_address"] = cached_avail["street_address"]
            
            # Fill in other location details from cache if missing
            for field in ["city", "county", "state", "country", "region"]:
                if cached_avail.get(field) and not new_avail.get(field):
                    merged_avail[field] = cached_avail[field]
            
            merged["availability"] = merged_avail
        
        # Merge dishes (combine lists avoiding duplicates)
        if "dishes" in cached_data:
            if "dishes" not in merged:
                merged["dishes"] = cached_data["dishes"]
            else:
                # Handle different dish data structures
                if isinstance(merged["dishes"], list) and isinstance(cached_data["dishes"], list):
                    merged_dishes = merged["dishes"].copy()
                    
                    # Add cached dishes not already in new data
                    cached_dish_names = {dish.get("dish_name", "").lower() for dish in cached_data["dishes"]}
                    new_dish_names = {dish.get("dish_name", "").lower() for dish in merged_dishes}
                    
                    for dish in cached_data["dishes"]:
                        dish_name = dish.get("dish_name", "").lower()
                        if dish_name and dish_name not in new_dish_names:
                            merged_dishes.append(dish)
                    
                    merged["dishes"] = merged_dishes
                elif isinstance(merged["dishes"], dict) and isinstance(cached_data["dishes"], dict):
                    # Handle explicit/visual dishes structure
                    merged_dishes = merged["dishes"].copy()
                    
                    # Merge explicitly mentioned dishes
                    if "explicitly_mentioned" in merged_dishes and "explicitly_mentioned" in cached_data["dishes"]:
                        new_dish_names = {d.get("dish_name", "").lower() for d in merged_dishes["explicitly_mentioned"]}
                        for dish in cached_data["dishes"]["explicitly_mentioned"]:
                            if dish.get("dish_name", "").lower() not in new_dish_names:
                                merged_dishes["explicitly_mentioned"].append(dish)
                    
                    # Merge visually shown dishes
                    if "visually_shown" in merged_dishes and "visually_shown" in cached_data["dishes"]:
                        new_dish_names = {d.get("dish_name", "").lower() for d in merged_dishes["visually_shown"]}
                        for dish in cached_data["dishes"]["visually_shown"]:
                            if dish.get("dish_name", "").lower() not in new_dish_names:
                                merged_dishes["visually_shown"].append(dish)
                    
                    merged["dishes"] = merged_dishes
        
        # Preserve new visual data if available
        if "visual_data" in new_data:
            # Keep new visual data but enhance with cached data if needed
            if "visual_data" in cached_data:
                # Check for missing visual data categories in new data
                for category in ["detected_objects", "scene_categories", "food_items"]:
                    if (category not in new_data["visual_data"] or not new_data["visual_data"][category]) and \
                       category in cached_data["visual_data"] and cached_data["visual_data"][category]:
                        if category not in merged["visual_data"]:
                            merged["visual_data"][category] = []
                        merged["visual_data"][category].extend(cached_data["visual_data"][category])
        elif "visual_data" in cached_data:
            # Use cached visual data if no new visual data available
            merged["visual_data"] = cached_data["visual_data"]
        
        # Combine activities lists
        if "activities" in cached_data and isinstance(cached_data["activities"], list):
            if "activities" not in merged or not merged["activities"]:
                merged["activities"] = cached_data["activities"]
            else:
                # Add unique activities from cache
                merged_activities = set(merged["activities"])
                for activity in cached_data["activities"]:
                    merged_activities.add(activity)
                merged["activities"] = list(merged_activities)
        
        # Combine vibes
        if cached_data.get("vibes") and not merged.get("vibes"):
            merged["vibes"] = cached_data["vibes"]
        
        return merged
    
    def cleanup_old_entries(self, days: int = None):
        """
        Remove cache entries older than specified days.
        
        Args:
            days: Days threshold (defaults to cache_lifetime_days if None)
        """
        if days is None:
            days = self.cache_lifetime_days
        
        threshold_date = datetime.now() - timedelta(days=days)
        formatted_date = threshold_date.strftime('%Y-%m-%d %H:%M:%S')
        
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        # Clean video cache
        cursor.execute('''
        DELETE FROM video_cache 
        WHERE last_accessed < ?
        ''', (formatted_date,))
        video_count = cursor.rowcount
        
        # Clean place cache
        cursor.execute('''
        DELETE FROM place_cache 
        WHERE last_accessed < ?
        ''', (formatted_date,))
        place_count = cursor.rowcount
        
        conn.commit()
        conn.close()
        
        print(f"Cleaned up {video_count} video cache entries and {place_count} place cache entries older than {days} days")
        
    def get_cache_stats(self) -> Dict[str, Any]:
        """
        Get statistics about the cache.
        
        Returns:
            Dictionary with cache statistics
        """
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        # Get video cache stats
        cursor.execute('SELECT COUNT(*) FROM video_cache')
        video_count = cursor.fetchone()[0]
        
        cursor.execute('SELECT MIN(created_at), MAX(created_at) FROM video_cache')
        video_date_range = cursor.fetchone()
        
        # Get place cache stats
        cursor.execute('SELECT COUNT(*) FROM place_cache')
        place_count = cursor.fetchone()[0]
        
        cursor.execute('SELECT COUNT(DISTINCT place_normalized) FROM place_cache')
        unique_places = cursor.fetchone()[0]
        
        cursor.execute('SELECT COUNT(DISTINCT country) FROM place_cache WHERE country IS NOT NULL AND country != ""')
        countries_count = cursor.fetchone()[0]
        
        cursor.execute('SELECT MIN(created_at), MAX(created_at) FROM place_cache')
        place_date_range = cursor.fetchone()
        
        # Get database size
        cursor.execute("PRAGMA page_count")
        page_count = cursor.fetchone()[0]
        cursor.execute("PRAGMA page_size")
        page_size = cursor.fetchone()[0]
        db_size = page_count * page_size / (1024 * 1024)  # Size in MB
        
        conn.close()
        
        return {
            "video_cache": {
                "count": video_count,
                "oldest": video_date_range[0] if video_date_range[0] else None,
                "newest": video_date_range[1] if video_date_range[1] else None,
            },
            "place_cache": {
                "count": place_count,
                "unique_places": unique_places,
                "countries": countries_count,
                "oldest": place_date_range[0] if place_date_range[0] else None,
                "newest": place_date_range[1] if place_date_range[1] else None,
            },
            "db_size_mb": round(db_size, 2),
            "cache_lifetime_days": self.cache_lifetime_days
        }

# Example usage function
def process_video_with_caching(url: str, agent_function, cache_system: VideoCachingSystem):
    """
    Process a video with caching support.
    
    Args:
        url: Video URL to process
        agent_function: Function that performs the actual video processing
        cache_system: VideoCachingSystem instance
        
    Returns:
        Analysis result dictionary
    """
    print(f"🔍 Processing video: {url}")
    
    # First, check if we have a cached result for this URL
    cached_result = cache_system.get_cached_video_result(url)
    if cached_result:
        print(f"✅ Found cached result for URL: {url}")
        return cached_result
    
    # If not cached, process the video
    print(f"⏳ No cached result found, processing video...")
    result = agent_function(url)
    
    # If processing succeeded, cache the result
    if result and not result.get("error"):
        # Before caching, merge with any cached place information
        merged_result = cache_system.merge_with_cached_places(result)
        
        # Cache the merged result
        cache_system.cache_video_result(url, merged_result)
        print(f"💾 Cached video analysis result for future use")
        
        return merged_result
    
    return result