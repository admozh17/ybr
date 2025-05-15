#!/usr/bin/env python3
"""
Simple script to associate existing content with admin user
"""

import sqlite3
import sys
from pathlib import Path

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

def associate_content_with_admin(db_path):
    """Associate all existing content with the admin user"""
    print(f"Associating content with admin user in {db_path}")
    
    try:
        conn = sqlite3.connect(db_path)
        cursor = conn.cursor()
        
        # First, check if admin user exists and get the ID
        cursor.execute("SELECT id FROM user WHERE username = ?", ("admin",))
        admin_user = cursor.fetchone()
        
        if not admin_user:
            print("Admin user not found. Creating admin user...")
            from werkzeug.security import generate_password_hash
            
            # Create admin user
            cursor.execute('''
                INSERT INTO user (email, username, password_hash, auth_provider, full_name)
                VALUES (?, ?, ?, ?, ?)
            ''', ('admin@example.com', 'admin', generate_password_hash('admin'), 'local', 'Admin User'))
            
            conn.commit()
            
            # Get the new admin ID
            cursor.execute("SELECT id FROM user WHERE username = ?", ("admin",))
            admin_user = cursor.fetchone()
        
        admin_id = admin_user[0]
        print(f"Admin user ID: {admin_id}")
        
        # Update result table
        cursor.execute("UPDATE result SET user_id = ? WHERE user_id IS NULL", (admin_id,))
        results_updated = cursor.rowcount
        print(f"Updated {results_updated} results")
        
        # Update album table
        cursor.execute("UPDATE album SET user_id = ? WHERE user_id IS NULL", (admin_id,))
        albums_updated = cursor.rowcount
        print(f"Updated {albums_updated} albums")
        
        # Commit changes
        conn.commit()
        print("Content association completed successfully")
        
        return True
    except sqlite3.Error as e:
        print(f"SQLite error: {e}")
        return False
    finally:
        if conn:
            conn.close()

if __name__ == "__main__":
    # Get database path from command line or search for it
    import os
    
    if len(sys.argv) > 1:
        db_path = sys.argv[1]
    else:
        db_path = find_database()
    
    if not db_path:
        print("Error: Could not find database file")
        print("Please specify the path to the database file:")
        print("python associate_content.py /path/to/your/database.db")
        sys.exit(1)
    
    if associate_content_with_admin(db_path):
        print("\n✅ Content association completed")
        print("\nAll existing content is now associated with the admin user:")
        print("  Email: admin@example.com")
        print("  Password: admin")
    else:
        print("\n❌ Content association failed")
