# debug_cache.py
from caching_system import VideoCachingSystem
import os

cache = VideoCachingSystem()

# Test with a few URL variants to check how they're hashed
test_urls = [
    "https://www.instagram.com/reel/DIe0l2Cvtrm/?igsh=MWNzamdydWR0bHB0bg",
    "https://www.instagram.com/reel/DIe0l2Cvtrm/",
    "https://instagram.com/reel/DIe0l2Cvtrm/",
    # Add the exact URL you've processed previously
]

print("Cache database path:", os.path.abspath(cache.db_path))
print("Cache database exists:", os.path.exists(cache.db_path))

for url in test_urls:
    url_hash = cache._hash_url(url)
    result = cache.get_cached_video_result(url)
    print(f"\nURL: {url}")
    print(f"Hash: {url_hash}")
    print(f"In cache: {result is not None}")
    
    if result:
        print(f"Result keys: {list(result.keys())}")
        print(f"Has activities: {len(result.get('activities', []))}")