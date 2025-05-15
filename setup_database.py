#!/usr/bin/env python3
"""
Script to add user_id columns and create admin user
"""

import sqlite3
import os
import sys
from werkzeug.security import generate_password_hash
from datetime import datetime

def find_database():
    """Find the database file in common locations"""
    # Common places to look
    possible_paths = [
        "instance/results.db",
        "../instance/results.db",
        "results.db",
        os.path.expanduser("~/ybr/instance/results.db")
    ]
    
    for path in possible_paths:
        if os.path.exists(path):
            print(f"Found database at: {path}")
            return path
    
    return None

def setup_database(db_path):
    """Add missing columns to database tables"""
    print(f"Setting up database at {db_path}")
    
    try:
        conn = sqlite3.connect(db_path)
        cursor = conn.cursor()
        
        # Check if result table has user_id column
        cursor.execute("PRAGMA table_info(result)")
        columns = cursor.fetchall()
        column_names = [col[1] for col in columns]
        
        if "user_id" not in column_names:
            print("Adding user_id column to result table...")
            cursor.execute("ALTER TABLE result ADD COLUMN user_id INTEGER")
            print("Column added successfully")
        else:
            print("user_id column already exists in result table")
        
        # Check if album table has user_id column
        cursor.execute("PRAGMA table_info(album)")
        columns = cursor.fetchall()
        column_names = [col[1] for col in columns]
        
        if "user_id" not in column_names:
            print("Adding user_id column to album table...")
            cursor.execute("ALTER TABLE album ADD COLUMN user_id INTEGER")
            print("Column added successfully")
        else:
            print("user_id column already exists in album table")
        
        # Check if user table exists
        cursor.execute("SELECT name FROM sqlite_master WHERE type='table' AND name='user'")
        if not cursor.fetchone():
            print("Creating user table...")
            cursor.execute('''
                CREATE TABLE user (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    email TEXT UNIQUE NOT NULL,
                    username TEXT UNIQUE NOT NULL,
                    password_hash TEXT,
                    is_active INTEGER DEFAULT 1,
                    profile_picture TEXT,
                    full_name TEXT,
                    auth_provider TEXT DEFAULT 'local',
                    provider_id TEXT,
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    last_login TIMESTAMP
                )
            ''')
            print("User table created successfully")
        else:
            print("User table already exists")
        
        # Create admin user
        print("Checking for admin user...")
        cursor.execute("SELECT * FROM user WHERE username = 'admin'")
        admin_user = cursor.fetchone()
        
        if not admin_user:
            print("Creating admin user...")
            now = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            cursor.execute('''
                INSERT INTO user (
                    email, username, password_hash, is_active,
                    full_name, auth_provider, created_at
                ) VALUES (?, ?, ?, ?, ?, ?, ?)
            ''', (
                'admin@example.com',
                'admin',
                generate_password_hash('admin'),
                1,
                'Admin User',
                'local',
                now
            ))
            print("Admin user created successfully")
            
            # Get the admin user ID
            cursor.execute("SELECT id FROM user WHERE username = 'admin'")
            admin_id = cursor.fetchone()[0]
        else:
            print("Admin user already exists")
            admin_id = admin_user[0]
        
        # Associate existing content with admin user
        cursor.execute("UPDATE result SET user_id = ? WHERE user_id IS NULL", (admin_id,))
        results_updated = cursor.rowcount
        print(f"Associated {results_updated} results with admin user")
        
        cursor.execute("UPDATE album SET user_id = ? WHERE user_id IS NULL", (admin_id,))
        albums_updated = cursor.rowcount
        print(f"Associated {albums_updated} albums with admin user")
        
        # Commit changes
        conn.commit()
        print("Database setup completed")
        
        return True
    except sqlite3.Error as e:
        print(f"SQLite error: {e}")
        return False
    finally:
        if conn:
            conn.close()

if __name__ == "__main__":
    # Get database path from command line or search for it
    if len(sys.argv) > 1:
        db_path = sys.argv[1]
    else:
        db_path = find_database()
    
    if not db_path:
        print("Error: Could not find database file")
        print("Please specify the path to the database file:")
        print("python setup_database.py /path/to/your/database.db")
        sys.exit(1)
    
    if setup_database(db_path):
        print("\n✅ Database setup completed successfully")
        print("\nYou can now log in with the admin user:")
        print("  Email: admin@example.com")
        print("  Username: admin")
        print("  Password: admin")
    else:
        print("\n❌ Database setup failed")
