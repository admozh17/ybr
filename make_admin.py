#!/usr/bin/env python3
"""Simple script to make a user an admin by directly accessing the database."""

import sqlite3
import sys
import os

def list_users(db_path="instance/results.db"):
    """List all users in the database."""
    # Check if database exists
    if not os.path.exists(db_path):
        print(f"Error: Database file not found at {db_path}")
        return False
    
    try:
        # Connect to the database
        conn = sqlite3.connect(db_path)
        cursor = conn.cursor()
        
        # Get all users
        cursor.execute("SELECT id, username, email FROM user ORDER BY id")
        users = cursor.fetchall()
        
        if not users:
            print("No users found in the database")
            conn.close()
            return False
        
        print("\nAvailable users:")
        print("ID | Username | Email")
        print("-" * 50)
        for user_id, username, email in users:
            print(f"{user_id} | {username} | {email}")
        
        conn.close()
        return True
        
    except sqlite3.Error as e:
        print(f"Database error: {e}")
        return False

def make_admin(email, db_path="instance/results.db"):
    """Make a user an admin by email address."""
    # Check if database exists
    if not os.path.exists(db_path):
        print(f"Error: Database file not found at {db_path}")
        return False
    
    try:
        # Connect to the database
        conn = sqlite3.connect(db_path)
        cursor = conn.cursor()
        
        # Check if is_admin column exists
        cursor.execute("PRAGMA table_info(user)")
        columns = [column[1] for column in cursor.fetchall()]
        
        # Add is_admin column if it doesn't exist
        if 'is_admin' not in columns:
            print("Adding is_admin column to user table...")
            cursor.execute("ALTER TABLE user ADD COLUMN is_admin BOOLEAN DEFAULT 0")
            conn.commit()
            print("Column added successfully")
        
        # Check if the user exists
        cursor.execute("SELECT id, username FROM user WHERE email = ?", (email,))
        user = cursor.fetchone()
        
        if not user:
            print(f"Error: No user found with email {email}")
            list_users(db_path)  # List available users
            conn.close()
            return False
        
        user_id, username = user
        
        # Check if already admin (only if column exists)
        if 'is_admin' in columns:
            cursor.execute("SELECT is_admin FROM user WHERE id = ?", (user_id,))
            is_admin = cursor.fetchone()[0]
            if is_admin:
                print(f"User {username} is already an admin")
                conn.close()
                return True
        
        # Update the user to be an admin
        cursor.execute("UPDATE user SET is_admin = 1 WHERE id = ?", (user_id,))
        conn.commit()
        
        print(f"Success: User {username} ({email}) is now an admin")
        conn.close()
        return True
        
    except sqlite3.Error as e:
        print(f"Database error: {e}")
        return False
    except Exception as e:
        print(f"Error: {e}")
        return False

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage:")
        print("  List all users: python make_admin.py --list")
        print("  Make user admin: python make_admin.py user@example.com")
        sys.exit(1)
    
    if sys.argv[1] == "--list":
        success = list_users()
    else:
        email = sys.argv[1]
        success = make_admin(email)
    
    sys.exit(0 if success else 1)