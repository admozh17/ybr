#!/usr/bin/env python3
"""
Database Migration Script - Add user_id to Result Table

This script modifies the Result table to add a user_id column,
fixing the 'no such column: result.user_id' error.
"""

import sqlite3
import os
import sys

def migrate_database(db_path="results.db"):
    """
    Add user_id column to Result table if it doesn't exist.
    
    Args:
        db_path: Path to the SQLite database file
    """
    print(f"Migrating database at {db_path}")
    
    # Check if database exists
    if not os.path.exists(db_path):
        print(f"Error: Database file {db_path} not found")
        return False
    
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
    # Get database path from command line argument or use default
    db_path = sys.argv[1] if len(sys.argv) > 1 else "results.db"
    
    if migrate_database(db_path):
        print("✅ Database migration successful")
    else:
        print("❌ Database migration failed")
