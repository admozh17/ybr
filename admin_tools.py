# Create a new file called admin_tools.py
import sys
import os
from web_app import app, db, User

def make_admin(email):
    with app.app_context():
        user = User.query.filter_by(email=email).first()
        if not user:
            print(f"Error: No user found with email {email}")
            return False
            
        user.is_admin = True
        db.session.commit()
        print(f"Success: User {user.username} ({email}) is now an admin")
        return True

if __name__ == "__main__":
    if len(sys.argv) != 2:
        print("Usage: python admin_tools.py user@example.com")
        sys.exit(1)
        
    email = sys.argv[1]
    success = make_admin(email)
    sys.exit(0 if success else 1)