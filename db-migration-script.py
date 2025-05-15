#!/usr/bin/env python3
"""
Database Migration Script - Add user_id to Result Table

This script modifies the Result table to add a user_id column,
fixing the 'no such column: result.user_id' error.
"""

import sqlite3
import os
import sys
import glob

def find_database():
    """
    Search for the SQLite database file in common locations.
    
    Returns:
        Path to the database file if found, None otherwise
    """
    # Common paths to search for the database
    possible_paths = [
        "results.db",                # Current directory
        "instance/results.db",       # Flask default instance folder
        "../instance/results.db",    # Parent directory instance folder
        "/tmp/results.db",           # Temporary directory
        os.path.expanduser("~/results.db")  # Home directory
    ]
    
    # Also search for any .db file in current and parent directories
    db_files = glob.glob("*.db") + glob.glob("../*.db") + glob.glob("instance/*.db")
    possible_paths.extend(db_files)
    
    # Remove duplicates
    possible_paths = list(set(possible_paths))
    
    print("Searching for database file...")
    for path in possible_paths:
        if os.path.exists(path):
            print(f"Found database at: {path}")
            # Verify it's a SQLite database file
            try:
                conn = sqlite3.connect(path)
                cursor = conn.cursor()
                # Check if it has a result table
                cursor.execute("SELECT name FROM sqlite_master WHERE type='table' AND name='result'")
                if cursor.fetchone():
                    conn.close()
                    return path
                conn.close()
            except sqlite3.Error:
                # Not a valid SQLite database or doesn't have our table
                pass
    
    return None

def migrate_database(db_path):
    """
    Add user_id column to Result table if it doesn't exist.
    
    Args:
        db_path: Path to the SQLite database file
    """
    print(f"Migrating database at {db_path}")
    
    try:
        # Connect to database
        conn = sqlite3.connect(db_path)
        cursor = conn.cursor()
        
        # Check if user_id column already exists
        cursor.execute("PRAGMA table_info(result)")
        columns = cursor.fetchall()
        column_names = [col[1] for col in columns]
        
        if "user_id" in column_names:
            print("user_id column already exists in Result table")
            conn.close()
            return True
        
        # Add user_id column to Result table
        print("Adding user_id column to Result table...")
        cursor.execute("ALTER TABLE result ADD COLUMN user_id INTEGER")
        
        # Commit changes and close connection
        conn.commit()
        conn.close()
        
        print("Migration completed successfully")
        return True
        
    except sqlite3.Error as e:
        print(f"SQLite error: {e}")
        return False

if __name__ == "__main__":
    # Get database path from command line argument or search for it
    if len(sys.argv) > 1:
        db_path = sys.argv[1]
        if not os.path.exists(db_path):
            print(f"Error: Database file {db_path} not found")
            sys.exit(1)
    else:
        db_path = find_database()
        if not db_path:
            print("Error: Could not find the database file.")
            print("Please specify the path to the database file as an argument:")
            print("python migrate_db.py /path/to/your/database.db")
            sys.exit(1)
    
    if migrate_database(db_path):
        print("✅ Database migration successful")
    else:
        print("❌ Database migration failed")