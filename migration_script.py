#!/usr/bin/env python3
"""
Migration script to populate the cache from existing database entries.
"""

import os
import sys
import argparse
import sqlite3
import json
import time
from caching_system import VideoCachingSystem

def migrate_results_to_cache(source_db_path, cache_db_path=None, batch_size=50):
    """
    Migrate existing results from the application database to the cache.
    
    Args:
        source_db_path: Path to the source SQLite database
        cache_db_path: Path to the cache database (uses default if None)
        batch_size: Number of records to process in each batch
    """
    # Initialize caching system
    cache = VideoCachingSystem(cache_db_path) if cache_db_path else VideoCachingSystem()
    
    # Connect to source database
    source_conn = sqlite3.connect(source_db_path)
    source_conn.row_factory = sqlite3.Row
    cursor = source_conn.cursor()
    
    # Get total count for progress tracking
    cursor.execute("SELECT COUNT(*) FROM result WHERE status = 'completed'")
    total_count = cursor.fetchone()[0]
    
    print(f"Found {total_count} completed results to migrate")
    
    # Process in batches
    offset = 0
    imported_count = 0
    skipped_count = 0
    error_count = 0
    
    start_time = time.time()
    
    while True:
        cursor.execute(
            """
            SELECT id, url, data, timestamp
            FROM result
            WHERE status = 'completed'
            ORDER BY id
            LIMIT ? OFFSET ?
            """,
            (batch_size, offset)
        )
        
        batch = cursor.fetchall()
        if not batch:
            break
        
        print(f"Processing batch of {len(batch)} results (offset: {offset})")
        
        for row in batch:
            try:
                result_id = row['id']
                url = row['url']
                
                # Parse JSON data
                try:
                    data = json.loads(row['data'])
                except (json.JSONDecodeError, TypeError):
                    print(f"❌ Error: Invalid JSON data for result {result_id}")
                    error_count += 1
                    continue
                
                # Skip if URL is missing
                if not url:
                    print(f"⚠️ Skipping result {result_id}: Missing URL")
                    skipped_count += 1
                    continue
                
                # Skip if already in cache
                if cache.get_cached_video_result(url):
                    print(f"⚠️ Skipping result {result_id}: Already in cache")
                    skipped_count += 1
                    continue
                
                # Add to cache
                success = cache.cache_video_result(url, data)
                
                if success:
                    imported_count += 1
                    print(f"✅ Imported result {result_id} for URL: {url}")
                else:
                    error_count += 1
                    print(f"❌ Failed to import result {result_id}")
                
            except Exception as e:
                error_count += 1
                print(f"❌ Error processing result {row['id']}: {str(e)}")
        
        # Update offset for next batch
        offset += batch_size
        
        # Print progress
        progress = min(100, (offset / total_count) * 100)
        elapsed = time.time() - start_time
        print(f"Progress: {progress:.1f}% ({offset}/{total_count}) - Elapsed: {elapsed:.1f}s")
    
    # Close database connection
    source_conn.close()
    
    # Print summary
    print("\n" + "=" * 50)
    print("Migration Summary:")
    print(f"Total records processed: {offset}")
    print(f"Successfully imported: {imported_count}")
    print(f"Skipped (already in cache/invalid): {skipped_count}")
    print(f"Errors: {error_count}")
    print(f"Total time: {time.time() - start_time:.1f} seconds")
    print("=" * 50)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Migrate existing results to cache")
    parser.add_argument("source_db", help="Path to source SQLite database")
    parser.add_argument("--cache-db", help="Path to cache database (uses default if not specified)")
    parser.add_argument("--batch-size", type=int, default=50, help="Batch size for processing")
    
    args = parser.parse_args()
    
    if not os.path.exists(args.source_db):
        print(f"Error: Source database not found at {args.source_db}")
        sys.exit(1)
    
    migrate_results_to_cache(args.source_db, args.cache_db, args.batch_size)
