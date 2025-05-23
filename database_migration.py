# Create this as a migration file or run it directly in your Flask shell

from flask import Flask
from flask_sqlalchemy import SQLAlchemy

# If running as a script, initialize your app context:
# with app.app_context():
#     db.create_all()

# Or create this as a Flask-Migrate migration:
"""
Flask-Migrate commands to run:

1. Initialize migrations (if not done):
   flask db init

2. Create migration:
   flask db migrate -m "Add social features - UserFriend table"

3. Apply migration:
   flask db upgrade
"""

# Manual SQL for the UserFriend table if needed:
CREATE_USER_FRIENDS_TABLE = """
CREATE TABLE IF NOT EXISTS user_friends (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    user_id INTEGER NOT NULL,
    friend_id INTEGER NOT NULL,
    status VARCHAR(20) DEFAULT 'pending',
    created_at DATETIME DEFAULT CURRENT_TIMESTAMP,
    FOREIGN KEY (user_id) REFERENCES user (id),
    FOREIGN KEY (friend_id) REFERENCES user (id),
    UNIQUE (user_id, friend_id)
);
"""

# If you need to add the missing columns to the User table:
ALTER_USER_TABLE = """
ALTER TABLE user ADD COLUMN is_admin BOOLEAN DEFAULT 0;
"""