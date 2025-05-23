#!/usr/bin/env python3
"""Flask web application for video analysis with waterfall pipeline architecture."""

# ── Core std-lib imports ───────────────────────────────────────
import asyncio, json, os, pathlib, tempfile, time
from datetime import datetime

# ── Flask & extensions (IMPORT *BEFORE* using Flask()) ─────────
from flask import (Flask, flash, jsonify, redirect, render_template,
                   request, url_for)
from flask_sqlalchemy import SQLAlchemy
from flask_migrate import Migrate
from flask_login import (LoginManager, UserMixin, current_user, login_required,
                         login_user, logout_user)
from authlib.integrations.flask_client import OAuth
from werkzeug.middleware.proxy_fix import ProxyFix
from werkzeug.security import check_password_hash, generate_password_hash
from sqlalchemy.orm.attributes import flag_modified
from dotenv import load_dotenv
import requests
from flask import abort  # Add at top if not already imported
from cache_adapter import check_cache_health





# ── Create app & DB objects ────────────────────────────────────
app = Flask(__name__)
app.config.from_object("config.Config")      # adjust as needed

# Load environment variables
load_dotenv()

# Add this context processor to make config available in templates
@app.context_processor
def inject_config():
    return {
        'config': {
            'GOOGLE_API_KEY': os.getenv('GOOGLE_API_KEY', ''),
            'GOOGLE_MAPS_API_KEY': os.getenv('GOOGLE_MAPS_API_KEY', ''),
            'GOOGLE_PLACES_API_KEY': os.getenv('GOOGLE_PLACES_API_KEY', ''),
        }
    }

# Enhanced photo route to use the Places API key specifically
@app.route("/photo/<path:photo_ref>")
def google_photo(photo_ref: str):
    """
    Streams a Google Places photo using the Places API key
    """
    # Use the specific Places API key for photos
    api_key = os.getenv("GOOGLE_PLACES_API_KEY") or os.getenv("GOOGLE_API_KEY")
    if not api_key:
        print("Error: No Google API key found for Places")
        return "Google API key not set", 500

 
    try:
        # 1. Ask Google for the photo redirect
        redirect_resp = requests.get(
            "https://maps.googleapis.com/maps/api/place/photo",
            params={
                "maxwidth": 400,
                "photoreference": photo_ref,
                "key": api_key,
            },
            allow_redirects=False,
            timeout=8,
        )

        if redirect_resp.status_code == 403:
            print(
                "API key unauthorized. Please check your Google Maps API key configuration"
            )
            return "API key unauthorized", 403

        if redirect_resp.status_code not in (302, 301):
            print(f"Photo API error: {redirect_resp.status_code}")
            return "Photo not found", 404

        # 2. Grab the actual JPEG the redirect points to
        image_resp = requests.get(redirect_resp.headers["Location"], timeout=8)

        return (
            image_resp.content,
            200,
            {"Content-Type": image_resp.headers.get("Content-Type", "image/jpeg")},
        )
    except Exception as e:
        print(f"Error fetching photo: {e}")
        return "Error fetching photo", 500

# Configure the app with database connection info
app.config["SQLALCHEMY_DATABASE_URI"] = "sqlite:///results.db"
app.config["SQLALCHEMY_TRACK_MODIFICATIONS"] = False
app.config["SECRET_KEY"] = os.getenv("SECRET_KEY", "dev-key-change-in-production")

# Handle proxy headers for proper URL generation (for OAuth callbacks)
app.wsgi_app = ProxyFix(app.wsgi_app, x_proto=1, x_host=1)

# Initialize database - IMPORTANT: Do this only once
db = SQLAlchemy(app)
migrate = Migrate(app, db)


from cache_adapter import cached_process_video, get_cache_stats, cleanup_cache

# ── Project-specific imports (after db exists) ─────────────────
from agent import Agent, PipelineStage, ExtractionState
from extractor import fetch_clip
from vector_manager import VectorManager

# Initialize login manager
login_manager = LoginManager()
login_manager.init_app(app)
login_manager.login_view = 'login'
login_manager.login_message_category = 'info'

# Initialize vector manager
vector_manager = VectorManager()

# Run cache health check on startup
print("\n=== Running Cache Health Check on Startup ===")
check_cache_health()
print("=== Cache Health Check Complete ===\n")

# In-memory cache for active extractions
active_extractions = {}


# Define models
# Fixed User model - replace the User class in web_app.py with this:

class User(db.Model, UserMixin):
    id = db.Column(db.Integer, primary_key=True)
    is_admin = db.Column(db.Boolean, default=False)
    email = db.Column(db.String(120), unique=True, nullable=False)
    username = db.Column(db.String(80), unique=True, nullable=False)
    password_hash = db.Column(db.String(256), nullable=True)
    is_active = db.Column(db.Boolean, default=True)
    
    # Profile information
    profile_picture = db.Column(db.String(500), nullable=True)
    full_name = db.Column(db.String(100), nullable=True)
    
    # OAuth related fields
    auth_provider = db.Column(db.String(20), nullable=True)  # 'local', 'google', 'instagram'
    provider_id = db.Column(db.String(100), nullable=True)
    
    # Timestamps
    created_at = db.Column(db.DateTime, default=datetime.utcnow)
    last_login = db.Column(db.DateTime, nullable=True)
    
    # Relationships
    results = db.relationship('Result', backref='user', lazy=True)
    albums = db.relationship('Album', backref='user', lazy=True)
    
    def set_password(self, password):
        self.password_hash = generate_password_hash(password)
        
    def check_password(self, password):
        if self.password_hash:
            return check_password_hash(self.password_hash, password)
        return False
    
    @classmethod
    def find_by_email(cls, email):
        return cls.query.filter_by(email=email).first()
    
    @classmethod
    def find_by_provider_id(cls, provider, provider_id):
        return cls.query.filter_by(auth_provider=provider, provider_id=provider_id).first()
    
    def update_last_login(self):
        self.last_login = datetime.utcnow()
        db.session.commit()

    def get_friends(self):
        """Get all accepted friends"""
        friend_ids = db.session.query(UserFriend.friend_id).filter(
            UserFriend.user_id == self.id,
            UserFriend.status == 'accepted'
        ).union(
            db.session.query(UserFriend.user_id).filter(
                UserFriend.friend_id == self.id,
                UserFriend.status == 'accepted'
            )
        ).all()
        
        friend_ids = [f[0] for f in friend_ids]
        return User.query.filter(User.id.in_(friend_ids)).all()

    def is_friend_with(self, user_id):
        """Check if users are friends"""
        return UserFriend.query.filter(
            ((UserFriend.user_id == self.id) & (UserFriend.friend_id == user_id)) |
            ((UserFriend.user_id == user_id) & (UserFriend.friend_id == self.id)),
            UserFriend.status == 'accepted'
        ).first() is not None

    def get_pending_friend_requests(self):
        """Get pending friend requests received by this user"""
        return UserFriend.query.filter_by(
            friend_id=self.id,
            status='pending'
        ).all()

    def get_sent_friend_requests(self):
        """Get pending friend requests sent by this user"""
        return UserFriend.query.filter_by(
            user_id=self.id,
            status='pending'
        ).all()


class UserFriend(db.Model):
    __tablename__ = 'user_friends'
    
    id = db.Column(db.Integer, primary_key=True)
    user_id = db.Column(db.Integer, db.ForeignKey('user.id'), nullable=False)
    friend_id = db.Column(db.Integer, db.ForeignKey('user.id'), nullable=False)
    status = db.Column(db.String(20), default='pending')  # pending, accepted, blocked
    created_at = db.Column(db.DateTime, default=datetime.utcnow)
    
    # Relationships
    user = db.relationship('User', foreign_keys=[user_id], backref='sent_friend_requests')
    friend = db.relationship('User', foreign_keys=[friend_id], backref='received_friend_requests')
    
    __table_args__ = (db.UniqueConstraint('user_id', 'friend_id', name='unique_friendship'),)
    

    


class Result(db.Model):
    """Persist one analysis run (video or URL)."""

    __tablename__ = "result"

    # ── Primary + foreign keys ──────────────────────────────────────────────
    id        = db.Column(db.Integer, primary_key=True)
    user_id   = db.Column(db.Integer, db.ForeignKey("user.id"), nullable=True)

    # ── Core payload ────────────────────────────────────────────────────────
    url       = db.Column(db.String(500), nullable=False)
    data      = db.Column(db.JSON,        nullable=False)
    timestamp = db.Column(
        db.DateTime,
        nullable=False,
        default=datetime.utcnow,
    )

    # ── Incremental-processing status (added for async waterfall UI) ───────
    status        = db.Column(db.String(20), default="pending")   # pending / processing / completed / error
    current_stage = db.Column(db.Integer,    default=0)           # 0-3 (we have 4 stages)
    progress      = db.Column(db.Float,      default=0.0)         # 0-100 percentage

    # ───────────────────────────────────────────────────────────────────────

    # ---------- Helper methods --------------------------------------------
    def has_visual_data(self) -> bool:
        """True if GPT-Vision or other Stage-2 models populated visual data."""
        return bool(self.data.get("recognition_data"))

    def get_visual_summary(self) -> dict | None:
        """
        Return a compact summary of visual-recognition facts
        (time, frames, models) or None if none exist.
        """
        if not self.has_visual_data():
            return None

        rec = self.data["recognition_data"]
        return {
            "processing_time" : rec.get("processing_time", 0),
            "frames_processed": rec.get("frames_processed", 0),
            "models_used"     : rec.get("models_used", []),
        }

    def to_dict(self) -> dict:
        """Serialize row for JSON responses / status polling."""
        return {
            "id"            : self.id,
            "url"           : self.url,
            "timestamp"     : self.timestamp.isoformat(),
            "status"        : self.status,
            "current_stage" : self.current_stage,
            "progress"      : self.progress,
            "data"          : self.data,
            "user_id"       : self.user_id,
        }

    # ---------- Representation --------------------------------------------
    def __repr__(self) -> str:
        return f"<Result id={self.id} status={self.status} progress={self.progress:.0f}%>"


class Album(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    name = db.Column(db.String(100), nullable=False)
    activities = db.Column(db.JSON, nullable=False, default=list)
    timestamp = db.Column(db.DateTime, default=datetime.utcnow)
    user_id = db.Column(db.Integer, db.ForeignKey('user.id'), nullable=True)



# 2. ADD THESE METHODS to your existing User class (after line 115, before the class ends):

    def get_friends(self):
        """Get all accepted friends"""
        friend_ids = db.session.query(UserFriend.friend_id).filter(
            UserFriend.user_id == self.id,
            UserFriend.status == 'accepted'
        ).union(
            db.session.query(UserFriend.user_id).filter(
                UserFriend.friend_id == self.id,
                UserFriend.status == 'accepted'
            )
        ).all()
        
        friend_ids = [f[0] for f in friend_ids]
        return User.query.filter(User.id.in_(friend_ids)).all()

    def is_friend_with(self, user_id):
        """Check if users are friends"""
        return UserFriend.query.filter(
            ((UserFriend.user_id == self.id) & (UserFriend.friend_id == user_id)) |
            ((UserFriend.user_id == user_id) & (UserFriend.friend_id == self.id)),
            UserFriend.status == 'accepted'
        ).first() is not None

@app.route("/api/friends/requests")
@login_required
def get_friend_requests():
    """Get pending friend requests for current user"""
    try:
        # Requests received by current user
        received_requests = UserFriend.query.filter_by(
            friend_id=current_user.id,
            status='pending'
        ).all()
        
        # Requests sent by current user
        sent_requests = UserFriend.query.filter_by(
            user_id=current_user.id,
            status='pending'
        ).all()
        
        received_data = []
        for request in received_requests:
            sender = db.session.get(User, request.user_id)
            if sender:
                received_data.append({
                    'id': request.id,
                    'sender': {
                        'id': sender.id,
                        'username': sender.username,
                        'full_name': sender.full_name,
                        'profile_picture': sender.profile_picture
                    },
                    'created_at': request.created_at.isoformat()
                })
        
        sent_data = []
        for request in sent_requests:
            recipient = db.session.get(User, request.friend_id)
            if recipient:
                sent_data.append({
                    'id': request.id,
                    'recipient': {
                        'id': recipient.id,
                        'username': recipient.username,
                        'full_name': recipient.full_name,
                        'profile_picture': recipient.profile_picture
                    },
                    'created_at': request.created_at.isoformat()
                })
        
        return jsonify({
            'received': received_data,
            'sent': sent_data
        })
        
    except Exception as e:
        app.logger.error(f"Error getting friend requests: {str(e)}")
        return jsonify({'error': str(e)}), 500


@app.route("/api/friends/decline/<int:request_id>", methods=["POST"])
@login_required
def decline_friend_request(request_id):
    """Decline friend request"""
    try:
        friend_request = UserFriend.query.get_or_404(request_id)
        
        if friend_request.friend_id != current_user.id:
            return jsonify({'error': 'Unauthorized'}), 403
        
        db.session.delete(friend_request)
        db.session.commit()
        
        return jsonify({'message': 'Friend request declined'})
        
    except Exception as e:
        app.logger.error(f"Error declining friend request: {str(e)}")
        return jsonify({'error': str(e)}), 500


@app.route("/api/friends/remove/<int:friend_id>", methods=["POST"])
@login_required
def remove_friend(friend_id):
    """Remove a friend"""
    try:
        # Find the friendship record
        friendship = UserFriend.query.filter(
            ((UserFriend.user_id == current_user.id) & (UserFriend.friend_id == friend_id)) |
            ((UserFriend.user_id == friend_id) & (UserFriend.friend_id == current_user.id)),
            UserFriend.status == 'accepted'
        ).first()
        
        if not friendship:
            return jsonify({'error': 'Friendship not found'}), 404
        
        db.session.delete(friendship)
        db.session.commit()
        
        return jsonify({'message': 'Friend removed successfully'})
        
    except Exception as e:
        app.logger.error(f"Error removing friend: {str(e)}")
        return jsonify({'error': str(e)}), 500


@app.route("/api/locations")
def get_locations():
    """Get available locations from analyzed results"""
    try:
        if current_user.is_authenticated:
            results = Result.query.filter_by(user_id=current_user.id).all()
        else:
            results = Result.query.filter_by(user_id=None).all()
        
        locations = set()
        
        for result in results:
            activities = result.data.get('activities', [])
            for activity in activities:
                availability = activity.get('availability', {})
                if availability.get('city'):
                    locations.add(availability['city'])
                if availability.get('country'):
                    locations.add(availability['country'])
                if availability.get('region'):
                    locations.add(availability['region'])
        
        # Convert to list of objects with id and name
        location_list = [{'id': loc.lower(), 'name': loc} for loc in sorted(locations)]
        
        return jsonify(location_list)
        
    except Exception as e:
        app.logger.error(f"Error getting locations: {str(e)}")
        return jsonify([])


@app.route("/album/<int:album_id>/preview")
def album_preview(album_id):
    """Get preview images for an album"""
    try:
        album = db.session.get(Album, album_id)
        if not album:
            return jsonify({'error': 'Album not found'}), 404
        
        image_urls = []
        
        for activity_ref in album.activities[:3]:  # Get first 3 images
            try:
                result_id, activity_index = activity_ref.split('-')
                result = Result.query.get(int(result_id))
                
                if result and result.data.get('activities'):
                    activity = result.data['activities'][int(activity_index)]
                    image_url = activity.get('image_url')
                    if image_url:
                        image_urls.append(image_url)
            except (ValueError, IndexError):
                continue
        
        return jsonify({'image_urls': image_urls})
        
    except Exception as e:
        app.logger.error(f"Error getting album preview: {str(e)}")
        return jsonify({'error': str(e)}), 500







@login_manager.user_loader
def load_user(user_id):
    return db.session.get(User, int(user_id))


# Initialize OAuth
oauth = OAuth(app)

# Set up Google OAuth
google = oauth.register(
    name='google',
    client_id=os.getenv('GOOGLE_OAUTH_CLIENT_ID'),
    client_secret=os.getenv('GOOGLE_OAUTH_CLIENT_SECRET'),
    server_metadata_url='https://accounts.google.com/.well-known/openid-configuration',
    client_kwargs={
        'scope': 'openid email profile',
        'token_endpoint_auth_method': 'client_secret_post',
        'prompt': 'select_account'
    },
    # Add these lines for ID token validation
    compliance_fix=lambda session: session.__setattr__('verify', False),  # Disable strict verification temporarily
    issuer='https://accounts.google.com'  # Set expected issuer explicitly
)

# Set up Instagram OAuth
instagram = oauth.register(
    name='instagram',
    client_id=os.getenv('INSTAGRAM_CLIENT_ID'),
    client_secret=os.getenv('INSTAGRAM_CLIENT_SECRET'),
    access_token_url='https://api.instagram.com/oauth/access_token',
    authorize_url='https://api.instagram.com/oauth/authorize',
    api_base_url='https://api.instagram.com/v1/',
    client_kwargs={'scope': 'user_profile,user_media'},
)


# Create database tables
with app.app_context():
    db.create_all()





# Album routes - kept the same
@app.route("/album/create", methods=["POST"])
@login_required
def create_album():
    data = request.get_json()
    album = Album(
        name=data["name"], 
        activities=data.get("activities", []),
        user_id=current_user.id if current_user.is_authenticated else None
    )
    db.session.add(album)
    db.session.commit()
    return jsonify({"id": album.id})


@app.route("/album/<int:album_id>/add", methods=["POST"])
@login_required
def add_to_album(album_id):

    album = db.session.get(Album, album_id)
    if not album:
        abort(404)
    
    # Check if user owns the album
    if album.user_id and album.user_id != current_user.id:
        return jsonify({"error": "You don't have permission to modify this album"}), 403
        
    data = request.get_json()
    new_activities = data.get("activities", [])

    current_activities = album.activities or []
    current_activities.extend(new_activities)
    album.activities = current_activities

    db.session.commit()
    return jsonify({"success": True})
@app.route("/albums_view")
def albums_view():
    """Render albums view page"""
    try:
        # Get all public albums
        albums = Album.query.filter_by(user_id=None).order_by(Album.timestamp.desc()).all()
        
        # If user is authenticated, include their private albums
        if current_user.is_authenticated:
            user_albums = Album.query.filter_by(user_id=current_user.id).order_by(Album.timestamp.desc()).all()
            # Combine lists (user's albums first)
            all_albums = user_albums + [a for a in albums if a not in user_albums]
        else:
            all_albums = albums
            
    except Exception as e:
        print(f"Warning: {e}")
        # Fall back to all albums
        all_albums = Album.query.order_by(Album.timestamp.desc()).all()
    
    return render_template("albums.html", albums=all_albums)


@app.route("/albums")
def get_albums():
    """Fixed albums endpoint that properly handles user authentication"""
    try:
        if current_user.is_authenticated:
            # Get user's albums first
            user_albums = Album.query.filter_by(user_id=current_user.id).order_by(Album.timestamp.desc()).all()
            
            # Then get public albums (those with user_id=None)
            public_albums = Album.query.filter_by(user_id=None).order_by(Album.timestamp.desc()).all()
            
            # Combine user albums first, then public albums
            all_albums = user_albums + public_albums
            
            print(f"✅ Found {len(user_albums)} user albums and {len(public_albums)} public albums")
        else:
            # Get only public albums for anonymous users
            all_albums = Album.query.filter_by(user_id=None).order_by(Album.timestamp.desc()).all()
            print(f"✅ Found {len(all_albums)} public albums for anonymous user")
        
        return jsonify([
            {
                "id": album.id,
                "name": album.name,
                "activities": album.activities or [],  # Ensure it's always a list
                "timestamp": album.timestamp.isoformat() if album.timestamp else None,
                "user_id": album.user_id,
                "is_public": album.user_id is None
            }
            for album in all_albums
        ])
        
    except Exception as e:
        print(f"❌ Error in get_albums: {e}")
        import traceback
        traceback.print_exc()
        return jsonify([]) 




@app.route("/album/<int:album_id>")
def view_album(album_id):
    album = db.session.get(Album, album_id)
    if not album:
        abort(404)
    
    # Check if album is private and user has access
    if album.user_id and (not current_user.is_authenticated or album.user_id != current_user.id):
        flash("You don't have permission to view this album", "danger")
        return redirect(url_for('index'))
        
    activities = []
    for activity_ref in album.activities:
        result_id, activity_index = activity_ref.split("-")
        result = Result.query.get(int(result_id))
        if result and result.data.get("activities"):
            activity = result.data["activities"][int(activity_index)]
            activities.append(activity)
    return render_template("album.html", album=album, activities=activities)


@app.route("/album/<int:album_id>", methods=["DELETE"])
@login_required
def delete_album(album_id):
    album = db.session.get(Album, album_id)
    if not album:
        abort(404)
    
    # Check if user owns the album
    if album.user_id and album.user_id != current_user.id:
        return jsonify({"error": "You don't have permission to delete this album"}), 403
        
    db.session.delete(album)
    db.session.commit()
    return jsonify({"success": True})


@app.route("/album/<int:album_id>/remove", methods=["POST"])
@login_required
def remove_from_album(album_id):
    album = db.session.get(Album, album_id)
    if not album:
        abort(404)
    
    # Check if user owns the album
    if album.user_id and album.user_id != current_user.id:
        return jsonify({"error": "You don't have permission to modify this album"}), 403
        
    data = request.get_json()
    activity_ref = data.get("activity_ref")

    if activity_ref in album.activities:
        album.activities.remove(activity_ref)
        flag_modified(album, "activities")
        db.session.commit()
        return jsonify({"success": True})

    return jsonify({"error": "Activity not found in album"}), 404


def fetch_place_image(place_name: str, activity: dict = None) -> str:
    """
    Look up a place in Google Places, grab the first photo_reference,
    and return the **local proxy URL** (/photo/<ref>).
    Falls back to the old Unsplash placeholder if nothing is found.
    """
    api_key = os.getenv("GOOGLE_API_KEY")
    if not api_key:
        print("Warning: GOOGLE_API_KEY not set")
        return "https://via.placeholder.com/400x300?text=No+API+Key"

    try:
        # Find the place ID + photos in one call
        find_resp = requests.get(
            "https://maps.googleapis.com/maps/api/place/findplacefromtext/json",
            params={
                "input": place_name,
                "inputtype": "textquery",
                "fields": "photos,formatted_address",
                "key": api_key,
            },
            timeout=8,
        ).json()

        if find_resp.get("status") != "OK":
            print(
                f"Places API error: {find_resp.get('status')} - {find_resp.get('error_message', 'No error message')}"
            )
            return "https://via.placeholder.com/400x300?text=No+API+Key"

        candidates = find_resp.get("candidates", [])
        if not candidates or "photos" not in candidates[0]:
            print(f"No photos found for: {place_name}")
            return "https://via.placeholder.com/400x300?text=No+Photos"

        # Store the address if available
        if "formatted_address" in candidates[0] and activity is not None:
            activity["address"] = candidates[0]["formatted_address"]

        photo_ref = candidates[0]["photos"][0]["photo_reference"]
        return f"/photo/{photo_ref}"

    except Exception as e:
        print(f"Error fetching place image for {place_name}: {e}")
        return f"https://source.unsplash.com/400x300/?{place_name.replace(' ', '+')}"
    
def calculate_distance(lat1, lng1, lat2, lng2):
    """Calculate distance between two points in kilometers using Haversine formula"""
    import math
    
    # Convert latitude and longitude from degrees to radians
    lat1, lng1, lat2, lng2 = map(math.radians, [lat1, lng1, lat2, lng2])
    
    # Haversine formula
    dlat = lat2 - lat1
    dlng = lng2 - lng1
    a = math.sin(dlat/2)**2 + math.cos(lat1) * math.cos(lat2) * math.sin(dlng/2)**2
    c = 2 * math.asin(math.sqrt(a))
    
    # Radius of earth in kilometers
    r = 6371
    
    return c * r

@app.route("/api/map/stats")
def get_map_stats():
    """Get statistics for the map"""
    try:
        if current_user.is_authenticated:
            results = Result.query.filter_by(user_id=current_user.id).all()
        else:
            results = Result.query.filter_by(user_id=None).all()
        
        total_places = 0
        total_visits = 0
        cities = set()
        genres = {}
        
        for result in results:
            activities = result.data.get('activities', [])
            total_visits += len(activities)
            
            for activity in activities:
                if activity.get('place_name') and activity.get('availability', {}).get('lat'):
                    total_places += 1
                    
                    city = activity.get('availability', {}).get('city')
                    if city:
                        cities.add(city)
                    
                    genre = activity.get('genre', 'restaurant')
                    genres[genre] = genres.get(genre, 0) + 1
        
        return jsonify({
            'total_places': total_places,
            'total_visits': total_visits,
            'cities_visited': len(cities),
            'top_genres': dict(sorted(genres.items(), key=lambda x: x[1], reverse=True)[:5])
        })
        
    except Exception as e:
        app.logger.error(f"Error getting map stats: {str(e)}")
        return jsonify({'error': str(e)}), 500


@app.route("/api/map/search")
def enhanced_map_search():
    """Enhanced search API for map with address-specific filtering and exact place matching"""
    try:
        # Get search parameters
        query = request.args.get('q', '').strip()
        scope = request.args.get('scope', 'mine')  # mine, friends, global, nearby
        lat = request.args.get('lat', type=float)
        lng = request.args.get('lng', type=float)
        radius = request.args.get('radius', 25, type=int)  # km
        
        # NEW: Specific place targeting parameters
        target_result_id = request.args.get('target_result_id')
        target_activity_index = request.args.get('target_activity_index', type=int)
        exact_match = request.args.get('exact_match', 'false').lower() == 'true'
        
        # Filters
        genre = request.args.get('genre', '')
        cuisine = request.args.get('cuisine', '')
        min_rating = request.args.get('min_rating', type=float)
        
        # Selection parameters
        selected_places = request.args.get('selected', '').split(',') if request.args.get('selected') else []
        album_id = request.args.get('album')
        highlight_place = request.args.get('highlight')
        
        print(f"🔍 Enhanced search: query='{query}', scope={scope}, target={target_result_id}:{target_activity_index}, exact={exact_match}")
        
        # NEW: If we have a specific target, find just that place
        if target_result_id and target_activity_index is not None and exact_match:
            return get_exact_place_for_map(target_result_id, target_activity_index)
        
        # Build results query based on scope
        results_query = Result.query
        
        if scope == 'mine':
            if current_user.is_authenticated:
                results_query = results_query.filter_by(user_id=current_user.id)
            else:
                results_query = results_query.filter_by(user_id=None)
                
        elif scope == 'friends':
            if current_user.is_authenticated:
                friends = current_user.get_friends()
                friend_ids = [f.id for f in friends]
                results_query = results_query.filter(Result.user_id.in_(friend_ids))
            else:
                return jsonify({'error': 'Login required for friends search', 'results': []}), 401
                
        elif scope == 'global':
            pass  # All public results
            
        elif scope == 'nearby':
            if not lat or not lng:
                return jsonify({'error': 'Location required for nearby search', 'results': []}), 400
        
        results = results_query.all()
        
        # Process all activities and apply filters
        places = []
        seen_places = {}  # Track by unique address identifier
        
        for result in results:
            activities = result.data.get('activities', [])
            for activity_index, activity in enumerate(activities):
                place_name = activity.get('place_name', '')
                if not place_name:
                    continue
                
                availability = activity.get('availability', {})
                place_lat = availability.get('lat')
                place_lng = availability.get('lon') or availability.get('lng')
                
                # Skip places without coordinates for nearby search
                if scope == 'nearby' and (not place_lat or not place_lng):
                    continue
                
                # Create activity reference
                activity_ref = f"{result.id}_{activity_index}"
                
                # Apply selection filters
                if selected_places and activity_ref not in selected_places:
                    alt_ref = f"{result.id}-{activity_index}"
                    if alt_ref not in selected_places:
                        continue
                        
                if album_id:
                    album = db.session.get(Album, int(album_id))
                    if album:
                        album_refs = album.activities or []
                        if f"{result.id}-{activity_index}" not in album_refs:
                            continue
                
                # ENHANCED: Apply text search filter with address matching
                if query:
                    searchable_fields = [
                        place_name.lower(),
                        activity.get('genre', '').lower(),
                        activity.get('cuisine', '').lower(),
                        activity.get('vibes', '').lower(),
                        availability.get('city', '').lower(),
                        availability.get('state', '').lower(),
                        availability.get('country', '').lower(),
                        availability.get('region', '').lower(),
                        availability.get('street_address', '').lower(),
                    ]
                    
                    # Add food items to search
                    food_items = activity.get('visual_data', {}).get('food_items', [])
                    for item in food_items:
                        searchable_fields.append(item.get('name', '').lower())
                    
                    searchable_text = ' '.join(filter(None, searchable_fields))
                    query_words = query.lower().split()
                    if not all(word in searchable_text for word in query_words):
                        continue
                
                # Apply other filters
                if genre and activity.get('genre', '').lower() != genre.lower():
                    continue
                if cuisine and activity.get('cuisine', '').lower() != cuisine.lower():
                    continue
                
                # Calculate distance for nearby search
                distance_km = None
                if scope == 'nearby' and lat and lng and place_lat and place_lng:
                    distance_km = calculate_distance(lat, lng, float(place_lat), float(place_lng))
                    if distance_km > radius:
                        continue
                
                # Calculate rating
                reviews = activity.get('user_reviews', [])
                avg_rating = 0.0
                if reviews:
                    avg_rating = sum(r.get('rating', 0) for r in reviews) / len(reviews)
                
                # Apply rating filter
                if min_rating and avg_rating < min_rating:
                    continue
                
                # NEW: Create unique identifier based on address + coordinates
                street_address = availability.get('street_address', '').strip()
                city = availability.get('city', '').strip()
                
                # Create a unique key based on precise location
                if place_lat and place_lng:
                    # Use coordinates rounded to 6 decimal places for uniqueness
                    coord_key = f"{round(float(place_lat), 6)},{round(float(place_lng), 6)}"
                    unique_key = f"{place_name.lower()}_{coord_key}"
                elif street_address:
                    unique_key = f"{place_name.lower()}_{street_address.lower()}"
                else:
                    unique_key = f"{place_name.lower()}_{city.lower()}"
                
                # NEW: Only keep the most recent visit for each unique place
                current_visit_date = result.timestamp
                if unique_key in seen_places:
                    if seen_places[unique_key]['visit_date'] < current_visit_date:
                        # This is a more recent visit, replace the old one
                        places = [p for p in places if p['unique_key'] != unique_key]
                    else:
                        # Skip this older visit
                        continue
                
                # Get user info for the place
                place_user = None
                if result.user_id:
                    place_user = db.session.get(User, result.user_id)
                
                # Check if highlighted
                is_highlighted = False
                if highlight_place:
                    possible_ids = [
                        f"{result.id}-{activity_index}",
                        f"{result.id}_{activity_index}",
                        str(result.id)
                    ]
                    is_highlighted = any(pid == highlight_place for pid in possible_ids)
                
                place_data = {
                    'id': activity_ref,
                    'result_id': result.id,
                    'activity_index': activity_index,
                    'name': place_name,
                    'genre': activity.get('genre', ''),
                    'cuisine': activity.get('cuisine', ''),
                    'street_address': street_address,
                    'city': city,
                    'state': availability.get('state', ''),
                    'country': availability.get('country', ''),
                    'latitude': float(place_lat) if place_lat else None,
                    'longitude': float(place_lng) if place_lng else None,
                    'distance_km': distance_km,
                    'rating': avg_rating,
                    'review_count': len(reviews),
                    'vibes': activity.get('vibes', ''),
                    'image_url': activity.get('image_url', ''),
                    'source_url': result.url,
                    'visit_date': current_visit_date.isoformat(),
                    'visual_data': activity.get('visual_data', {}),
                    'is_highlighted': is_highlighted,
                    'user': {
                        'id': result.user_id,
                        'username': place_user.username if place_user else 'Anonymous',
                        'profile_picture': place_user.profile_picture if place_user else None,
                        'is_friend': current_user.is_friend_with(result.user_id) if current_user.is_authenticated and result.user_id else False
                    } if result.user_id else None,
                    'scope': scope,
                    'full_address': format_full_address(availability),
                    'unique_key': unique_key  # For internal tracking
                }
                
                places.append(place_data)
                seen_places[unique_key] = {
                    'visit_date': current_visit_date,
                    'place_data': place_data
                }
        
        # Sort results
        sort_by = request.args.get('sort', 'relevance')
        if sort_by == 'distance' and scope == 'nearby':
            places.sort(key=lambda x: x['distance_km'] or 999)
        elif sort_by == 'rating':
            places.sort(key=lambda x: x['rating'], reverse=True)
        elif sort_by == 'recent':
            places.sort(key=lambda x: x['visit_date'], reverse=True)
        elif sort_by == 'name':
            places.sort(key=lambda x: x['name'].lower())
        elif sort_by == 'relevance':
            if query:
                places.sort(key=lambda x: x['name'].lower())
            else:
                places.sort(key=lambda x: x['rating'], reverse=True)
        
        # Clean up the unique_key from response
        for place in places:
            place.pop('unique_key', None)
        
        # Limit results
        limit = request.args.get('limit', 50, type=int)
        places = places[:limit]
        
        print(f"✅ Found {len(places)} unique places matching search criteria")
        
        return jsonify({
            'query': query,
            'scope': scope,
            'search_type': 'enhanced_with_address_deduplication',
            'location': {'lat': lat, 'lng': lng, 'radius': radius} if lat and lng else None,
            'results': places,
            'total': len(places),
            'filters': {
                'genre': genre,
                'cuisine': cuisine,
                'min_rating': min_rating,
                'selected_count': len(selected_places),
                'album_id': album_id,
                'highlight': highlight_place
            }
        })
        
    except Exception as e:
        app.logger.error(f"Error in enhanced map search: {str(e)}")
        import traceback
        traceback.print_exc()
        return jsonify({'error': str(e), 'results': []}), 500


# 7. ADD THIS ROUTE for getting popular searches:

@app.route("/api/map/popular-searches")
def get_popular_searches():
    """Get popular search terms and categories"""
    try:
        scope = request.args.get('scope', 'global')
        
        # Build results query based on scope
        results_query = Result.query
        if scope == 'mine' and current_user.is_authenticated:
            results_query = results_query.filter_by(user_id=current_user.id)
        elif scope == 'friends' and current_user.is_authenticated:
            friends = current_user.get_friends()
            friend_ids = [f.id for f in friends]
            results_query = results_query.filter(Result.user_id.in_(friend_ids))
        
        results = results_query.all()
        
        # Analyze popular terms
        genres = {}
        cuisines = {}
        cities = {}
        food_items = {}
        
        for result in results:
            activities = result.data.get('activities', [])
            for activity in activities:
                # Count genres
                genre = activity.get('genre', '')
                if genre:
                    genres[genre] = genres.get(genre, 0) + 1
                
                # Count cuisines
                cuisine = activity.get('cuisine', '')
                if cuisine:
                    cuisines[cuisine] = cuisines.get(cuisine, 0) + 1
                
                # Count cities
                city = activity.get('availability', {}).get('city', '')
                if city:
                    cities[city] = cities.get(city, 0) + 1
                
                # Count food items
                food_list = activity.get('visual_data', {}).get('food_items', [])
                for food in food_list:
                    name = food.get('name', '')
                    if name:
                        food_items[name] = food_items.get(name, 0) + 1
        
        return jsonify({
            'scope': scope,
            'popular': {
                'genres': sorted(genres.items(), key=lambda x: x[1], reverse=True)[:10],
                'cuisines': sorted(cuisines.items(), key=lambda x: x[1], reverse=True)[:10],
                'cities': sorted(cities.items(), key=lambda x: x[1], reverse=True)[:10],
                'food_items': sorted(food_items.items(), key=lambda x: x[1], reverse=True)[:10]
            }
        })
        
    except Exception as e:
        app.logger.error(f"Error getting popular searches: {str(e)}")
        return jsonify({'error': str(e)}), 500


@app.route("/")
def index():
    if current_user.is_authenticated:
        # Get user's results
        results = Result.query.filter_by(user_id=current_user.id).order_by(Result.timestamp.desc()).all()
    else:
        # Get public results (those without user_id)
        results = Result.query.filter_by(user_id=None).order_by(Result.timestamp.desc()).all()
        
    return render_template("index.html", previous_results=results)


# Update the analyze endpoint to support incremental updates
@app.route("/analyze", methods=["POST"])
def analyze():
    url = request.form.get("url")
    enable_visual = request.form.get("enable_visual", "false").lower() == "true"
    
    if not url:
        return jsonify({"error": "URL is required", "activities": []})

    try:
        # Create initial result entry
        db_result = Result(
            url=url, 
            data={"url": url, "status": "processing"},
            user_id=current_user.id if current_user.is_authenticated else None,
            status="processing",
            current_stage=0,
            progress=0.0
        )
        db.session.add(db_result)
        db.session.commit()
        
        # Create a task ID for this processing job
        task_id = f"task_{db_result.id}"
        
        # Start processing in a separate thread
        app.logger.info(f"Starting processing task {task_id} for URL {url}")
        
        # Run this in a background thread
        import threading
        processing_thread = threading.Thread(
            target=process_video_in_background,
            args=(url, db_result.id, enable_visual)
        )
        processing_thread.daemon = True
        processing_thread.start()
        
        # Return immediate response with task ID
        return jsonify({
            "id": db_result.id,
            "url": url,
            "status": "processing",
            "task_id": task_id,
            "message": "Processing started"
        })
    except Exception as e:
        app.logger.error(f"Error starting processing for URL {url}: {str(e)}")
        return jsonify(
            {"error": f"Failed to start processing: {str(e)}", "activities": []}
        )


def process_video_in_background(url, result_id, enable_visual):
    """Process video in a background thread and update the database incrementally."""
    try:
        # Get the result from database
        with app.app_context():
            db_result = db.session.get(Result, result_id)
            if not db_result:
                app.logger.error(f"Result {result_id} not found")
                return
            
            # Create agent
            agent = Agent(use_gpu=True)
            
            # Define a wrapper function for the processing pipeline
            def process_video_pipeline(video_url, enable_visual_recognition=False):
                """Wrapper for the Agent's video processing pipeline"""
                # Create initial state
                state = ExtractionState()
                state.url = video_url
                state.started_at = time.time()
                
                with tempfile.TemporaryDirectory() as tmp:
                    tmp_path = pathlib.Path(tmp)
                    clip_path = tmp_path / "clip.mp4"
                    
                    # Step 1: Download video
                    try:
                        app.logger.info(f"Downloading video from {video_url}...")
                        # Update database - Stage 0 starting
                        with app.app_context():
                            db_result = db.session.get(Result, result_id)
                            if db_result:
                                db_result.current_stage = 0
                                db_result.progress = 5.0
                                db.session.commit()
                                
                        download_start = time.time()
                        fetch_clip(video_url, clip_path)
                        download_time = time.time() - download_start
                        state.performance_profile["download_time"] = download_time
                        
                        # Update database - Download completed
                        with app.app_context():
                            db_result = db.session.get(Result, result_id)
                            if db_result:
                                db_result.progress = 15.0
                                db.session.commit()
                        
                        # Store clip path in state
                        state.clip_path = clip_path
                    except Exception as e:
                        app.logger.error(f"Error downloading video: {str(e)}")
                        state.error = f"Download error: {str(e)}"
                        return {"error": str(e), "url": video_url}
                    
                    # Run each pipeline stage synchronously in sequence
                    loop = asyncio.new_event_loop()
                    asyncio.set_event_loop(loop)
                    
                    # Define stage progress mappings
                    stage_progress = {
                        0: (15, 25),   # Stage 0: 15-25%
                        1: (25, 60),   # Stage 1: 25-60%
                        2: (60, 80),   # Stage 2: 60-80%
                        3: (80, 95)    # Stage 3: 80-95%
                    }
                    
                    # Run all stages in sequence
                    stages = [
                        agent.stage0_metadata,
                        agent.stage1_basic,
                        agent.stage2_vision,
                        agent.stage3_fusion
                    ]
                    
                    for stage_idx, stage_func in enumerate(stages):
                        try:
                            # Update database - Stage starting
                            with app.app_context():
                                db_result = db.session.get(Result, result_id)
                                if db_result:
                                    db_result.current_stage = stage_idx
                                    db_result.progress = stage_progress[stage_idx][0]
                                    db.session.commit()
                            
                            # Run the stage
                            state = loop.run_until_complete(stage_func(state))
                            
                            # Update database - Stage completed
                            with app.app_context():
                                db_result = db.session.get(Result, result_id)
                                if db_result:
                                    db_result.progress = stage_progress[stage_idx][1]
                                    db.session.commit()
                            
                            if state.error:
                                break
                        except Exception as e:
                            app.logger.error(f"Error in stage {stage_idx}: {str(e)}")
                            state.error = f"Stage {stage_idx} error: {str(e)}"
                            break
                    
                    # Return final state as dictionary
                    return state.to_dict()
            
            # Use cached processing
            result = cached_process_video(url, process_video_pipeline, enable_visual)
            
            # Handle cache hits immediately
            if result.get("_cache", {}).get("cache_hit", False):
                app.logger.info(f"🎯 Using cached result for URL: {url}")

                # Add image URLs to cached result
                for activity in result.get("activities", []):
                    activity["image_url"] = fetch_place_image(
                        activity.get("place_name", ""), activity
                    )

                # Update database with enriched cached result
                db_result.data = result
                db_result.status = "completed"
                db_result.progress = 100.0
                db_result.current_stage = 3
                db.session.commit()

                # Add to vector database if not already there
                try:
                    vector_manager.update_result(str(db_result.id), result, url)
                    app.logger.info(f"Updated vector index for cached result {db_result.id}")
                except Exception as e:
                    app.logger.error(f"Error updating search index: {str(e)}")

                return

            
            # For cache misses, or after processing, continue with normal updates
            # Add images to activities
            for activity in result.get("activities", []):
                activity["image_url"] = fetch_place_image(
                    activity.get("place_name", ""), activity
                )
            
            # Update database with final results
            db_result.data = result
            db_result.status = "completed"
            db_result.progress = 100.0
            db_result.current_stage = 3
            db.session.commit()
            
            # Add to vector database
            try:
                vector_manager.add_result(str(db_result.id), result, url)
                app.logger.info(f"Added result {db_result.id} to semantic search index")
            except Exception as e:
                app.logger.error(f"Error adding to search index: {str(e)}")
                
            app.logger.info(f"Processing completed for URL {url}")
            
    except Exception as e:
        app.logger.error(f"Critical error in background processing: {str(e)}")
        import traceback
        traceback.print_exc()
        with app.app_context():
            db_result = db.session.get(Result, result_id)
            if db_result:
                db_result.status = "error"
                db_result.data = {"error": f"Processing error: {str(e)}", "url": url}
                db.session.commit()

# Add an endpoint to check the status of an analysis
@app.route("/analysis_status/<int:result_id>")
def analysis_status(result_id):
    """Get the current status of an analysis."""
    result = db.session.get(Result, result_id)
    if not result:
        return jsonify({"error": "Result not found"}), 404
    
    # Check if result is private and user has access
    if result.user_id and (not current_user.is_authenticated or result.user_id != current_user.id):
        return jsonify({"error": "You don't have permission to view this result"}), 403
    

    
    # Check for "pending forever" issue - detect if result has been stuck at the same progress
    # for too long and has not updated in over 3 minutes
    if result.status == "processing":
        # Check if we have a timestamp for the last progress update
        last_update = result.data.get("last_progress_update")
        current_time = datetime.utcnow().timestamp()
        
        if last_update and (current_time - last_update > 180):  # 3 minutes timeout
            # The process appears to be stuck - update to error state
            result.status = "error"
            result.data["error"] = "Processing timed out after 3 minutes of inactivity"
            db.session.commit()
    
    response = {
        "id": result.id,
        "url": result.url,
        "status": result.status,
        "current_stage": result.current_stage,
        "progress": result.progress
    }
    
    # If completed, include the full result
    if result.status == "completed":
        response["data"] = result.data
        
        # Check if this was a cache hit - add a message if it was
        if result.data.get("_cache", {}).get("cache_hit", True):
            response["cache_hit"] = True
            response["message"] = "Retrieved from cache"
            
    # If error, include the error message
    elif result.status == "error":
        response["error"] = result.data.get("error", "Unknown error")
        
    # If processing, include partial data if available
    elif result.status == "processing":
        # Include stage-specific data
        response["stage_data"] = {}
        if result.current_stage >= 1:  # After stage 0
            response["stage_data"]["caption_text"] = result.data.get("caption_text", "")
        if result.current_stage >= 2:  # After stage 1
            response["stage_data"]["speech_text"] = result.data.get("speech_text", "")
            response["stage_data"]["num_frames"] = result.data.get("num_frames", 0)
        if result.current_stage >= 3:  # After stage 2
            response["stage_data"]["frame_text"] = result.data.get("frame_text", "")
            response["stage_data"]["visual_results"] = result.data.get("visual_results", [])
            
        # Update last progress timestamp
        result.data["last_progress_update"] = datetime.utcnow().timestamp()
        db.session.commit()
    
    return jsonify(response)


@app.route("/gallery")
def gallery():
    if current_user.is_authenticated:
        # Get user's results
        results = Result.query.filter_by(user_id=current_user.id).order_by(Result.timestamp.desc()).all()
    else:
        # Get public results (those without user_id)
        results = Result.query.filter_by(user_id=None).order_by(Result.timestamp.desc()).all()
        
    return render_template("gallery.html", results=results)


@app.route(
    "/result/<int:result_id>/activity/<int:activity_index>",
    methods=["GET", "PUT", "DELETE"],
)
def manage_activity(result_id, activity_index):
    result = db.session.get(Result, result_id)
    if not result:
        return jsonify({"error": "Result not found"}), 404
    
    # Check if result is private and user has access
    if result.user_id and (not current_user.is_authenticated or result.user_id != current_user.id):
        if request.method == "GET":
            flash("You don't have permission to view this result", "danger")
            return redirect(url_for('index'))
        else:
            return jsonify({"error": "You don't have permission to modify this result"}), 403
    
    activities = result.data.get("activities", [])

    if activity_index >= len(activities):
        return jsonify({"error": "Activity not found"}), 404

    if request.method == "GET":
        return jsonify(activities[activity_index])

    elif request.method == "PUT":
        try:
            new_activity = request.json
            activities[activity_index] = new_activity
            result.data["activities"] = activities

            flag_modified(result, "data")  # ← force SQLAlchemy to update JSON
            db.session.commit()
            # Update vector database to reflect changes
            try:
                vector_manager.update_result(str(result_id), result.data, result.url)
                print(f"✅ Updated result {result_id} in search index")
            except Exception as e:
                print(f"❌ Error updating search index: {str(e)}")

            return jsonify({"success": True, "updated_data": new_activity})
        except Exception as e:
            db.session.rollback()
            return jsonify({"error": str(e)}), 500

    elif request.method == "DELETE":
        activities.pop(activity_index)
        result.data["activities"] = activities
        flag_modified(result, "data")

        db.session.commit()
        # Update vector database
        try:
            if activities:  # If there are still activities, update the vector
                vector_manager.update_result(str(result_id), result.data, result.url)
                print(f"✅ Updated result {result_id} in search index after activity deletion")
            else:  # If all activities are deleted, remove from vector database
                vector_manager.delete_result(str(result_id))
                print(f"✅ Removed result {result_id} from search index - all activities deleted")
        except Exception as e:
            print(f"❌ Error updating search index: {str(e)}")

        return jsonify({"success": True})


@app.route("/details/<int:result_id>/<int:activity_index>")
def details(result_id, activity_index=0):
    result = db.session.get(Result, result_id)
    if not result:
        return "Result not found", 404
    
    # Check if result is private and user has access
    if result.user_id and (not current_user.is_authenticated or result.user_id != current_user.id):
        flash("You don't have permission to view this result", "danger")
        return redirect(url_for('index'))
    
    activities = result.data.get("activities", [])
    if activity_index >= len(activities):
        return "Activity not found", 404
    activity = activities[activity_index]
    return render_template(
        "details.html",
        activity=activity,
        result_id=result_id,
        activity_index=activity_index,
        total_activities=len(activities),
    )
@app.route("/admin/cache-stats")
@login_required
def cache_statistics():
    """View cache statistics dashboard."""
    # Basic auth check - only allow admin users
    if not current_user.is_authenticated or not getattr(current_user, 'is_admin', False):
        flash("You don't have permission to access this page", "danger")
        return redirect(url_for('index'))
        
    # Get cache statistics
    stats = get_cache_stats()
    
    # Add a formatted version of DB size
    if stats["db_size_mb"] < 1:
        stats["formatted_size"] = f"{stats['db_size_mb'] * 1024:.2f} KB"
    elif stats["db_size_mb"] < 1000:
        stats["formatted_size"] = f"{stats['db_size_mb']:.2f} MB"
    else:
        stats["formatted_size"] = f"{stats['db_size_mb'] / 1024:.2f} GB"
    
    return render_template("admin/cache_stats.html", stats=stats)
@app.route("/admin/cache-maintenance", methods=["POST"])
@login_required
def cache_maintenance():
    """Perform cache maintenance operations."""
    # Basic auth check - only allow admin users
    if not current_user.is_authenticated or not getattr(current_user, 'is_admin', False):
        return jsonify({"error": "Unauthorized"}), 403
    
    action = request.form.get("action")
    
    if action == "cleanup":
        days = request.form.get("days")
        days = int(days) if days and days.isdigit() else None
        
        cleanup_cache(days)
        flash(f"Cache cleanup completed for entries older than {days or 'default'} days", "success")
    
    return redirect(url_for('cache_statistics'))

def ensure_activity_image(activity: dict):
    """Attach image_url if it's missing (used for legacy rows)."""
    if not activity.get("image_url"):
        activity["image_url"] = fetch_place_image(
            activity.get("place_name", ""), activity
        )


@app.route("/search")
def search_page():
    """Render search page"""
    return render_template("search.html")


@app.route("/api/search")
def search_api():
    """API endpoint for semantic search"""
    query = request.args.get("q", "")
    if not query:
        return jsonify({"error": "Query is required", "results": []})
    
    # Get filters from request
    filters = {}
    for key in ['genre', 'city', 'country', 'region']:
        if value := request.args.get(key):
            filters[key] = value
    
    # New visual filters
    visual_filters = []
    if object_filter := request.args.get('object'):
        visual_filters.append(('detected_objects', 'label', object_filter))
    if scene_filter := request.args.get('scene'):
        visual_filters.append(('scene_categories', 'category', scene_filter))
    
    # Perform search
    try:
        results = vector_manager.search(query, limit=20, filters=filters)
        
        # Apply visual filters (post-processing)
        if visual_filters:
            filtered_results = []
            for result in results:
                result_id = result["id"]
                db_result = db.session.get(Result, result_id)
                if db_result:
                    # Check if matches visual filters
                    matches_filters = True
                    for activity in db_result.data.get("activities", []):
                        if "visual_data" in activity:
                            for filter_type, filter_field, filter_value in visual_filters:
                                found = False
                                for item in activity["visual_data"].get(filter_type, []):
                                    if filter_value.lower() in item.get(filter_field, "").lower():
                                        found = True
                                        break
                                if not found:
                                    matches_filters = False
                                    break
                    
                    if matches_filters:
                        filtered_results.append(result)
            results = filtered_results
        
        # Enhance results with additional info
        for result in results:
            result_id = result["id"]
            db_result = db.session.get(Result, result_id)
            if db_result:
                # Add URL
                result["url"] = db_result.url
                
                # Add first activity image
                activities = db_result.data.get("activities", [])
                if activities and "image_url" in activities[0]:
                    result["image_url"] = activities[0]["image_url"]
                elif activities:
                    # Generate image URL if not already present
                    place_name = activities[0].get("place_name", "")
                    result["image_url"] = fetch_place_image(place_name)
                
                # Add visual data highlights
                result["visual_highlights"] = []
                for activity in activities:
                    if "visual_data" in activity:
                        # Add top scenes
                        for scene in activity["visual_data"].get("scene_categories", [])[:2]:
                            result["visual_highlights"].append({
                                "type": "scene",
                                "label": scene["category"],
                                "confidence": scene["confidence"]
                            })
        
        return jsonify({
            "query": query,
            "filters": filters,
            "visual_filters": [f[2] for f in visual_filters],
            "results": results
        })
    except Exception as e:
        app.logger.error(f"Search error: {str(e)}")
        return jsonify({"error": str(e), "results": []})


@app.route("/api/similar/<int:result_id>")
def similar_content(result_id):
    """API endpoint for finding similar content"""
    try:
        similar = vector_manager.get_similar_content(str(result_id), limit=5)
        
        # Enhance results with additional info
        for result in similar:
            db_result_id = result["id"]
            db_result = Result.query.get(db_result_id)
            if db_result:
                # Add URL
                result["url"] = db_result.url
                
                # Add first activity data
                activities = db_result.data.get("activities", [])
                if activities:
                    result["activity"] = activities[0]
        
        return jsonify({
            "result_id": result_id,
            "similar": similar
        })
    except Exception as e:
        app.logger.error(f"Similar content error: {str(e)}")
        return jsonify({"error": str(e), "similar": []})


# Authentication routes
@app.route('/login', methods=['GET', 'POST'])
def login():
    if current_user.is_authenticated:
        return redirect(url_for('index'))
        
    if request.method == 'POST':
        email = request.form.get('email')
        password = request.form.get('password')
        remember = 'remember' in request.form
        
        user = User.find_by_email(email)
        
        if user and user.check_password(password):
            login_user(user, remember=remember)
            user.update_last_login()
            
            next_page = request.args.get('next')
            return redirect(next_page or url_for('index'))
        else:
            flash('Invalid email or password', 'danger')
    
    return render_template('auth/login.html')


@app.route('/register', methods=['GET', 'POST'])
def register():
    if current_user.is_authenticated:
        return redirect(url_for('index'))
        
    if request.method == 'POST':
        email = request.form.get('email')
        username = request.form.get('username')
        password = request.form.get('password')
        confirm_password = request.form.get('confirm_password')
        
        # Validate input
        if password != confirm_password:
            flash('Passwords do not match', 'danger')
            return render_template('auth/register.html')
            
        existing_user = User.find_by_email(email)
        if existing_user:
            flash('Email already registered', 'danger')
            return render_template('auth/register.html')
            
        existing_username = User.query.filter_by(username=username).first()
        if existing_username:
            flash('Username already taken', 'danger')
            return render_template('auth/register.html')
            
        # Create new user
        user = User(
            email=email,
            username=username,
            auth_provider='local'
        )
        user.set_password(password)
        
        db.session.add(user)
        db.session.commit()
        
        flash('Registration successful! Please log in.', 'success')
        return redirect(url_for('login'))
        
    return render_template('auth/register.html')


@app.route('/logout')
@login_required
def logout():
    logout_user()
    flash('You have been logged out', 'info')
    return redirect(url_for('index'))


@app.route('/profile')
@login_required
def profile():
    return render_template('auth/profile.html', user=current_user)


@app.route('/profile/edit', methods=['GET', 'POST'])
@login_required
def edit_profile():
    if request.method == 'POST':
        username = request.form.get('username')
        full_name = request.form.get('full_name')
        
        # Check if username is available
        if username != current_user.username:
            existing_user = User.query.filter_by(username=username).first()
            if existing_user:
                flash('Username already taken', 'danger')
                return render_template('auth/edit_profile.html')
        
        # Update user
        current_user.username = username
        current_user.full_name = full_name
        
        # Handle password change if provided
        current_password = request.form.get('current_password')
        new_password = request.form.get('new_password')
        confirm_password = request.form.get('confirm_password')
        
        if current_password and new_password:
            if not current_user.check_password(current_password):
                flash('Current password is incorrect', 'danger')
                return render_template('auth/edit_profile.html')
                
            if new_password != confirm_password:
                flash('New passwords do not match', 'danger')
                return render_template('auth/edit_profile.html')
                
            current_user.set_password(new_password)
        
        db.session.commit()
        flash('Profile updated successfully', 'success')
        return redirect(url_for('profile'))
        
    return render_template('auth/edit_profile.html')


# Google OAuth routes
@app.route('/login/google')
def google_login():
    try:
        # Print OAuth configuration (but mask part of the secret)
        client_id = os.getenv('GOOGLE_OAUTH_CLIENT_ID')
        client_secret = os.getenv('GOOGLE_OAUTH_CLIENT_SECRET')
        if client_secret and len(client_secret) > 8:
            masked_secret = client_secret[:4] + '****' + client_secret[-4:]
        else:
            masked_secret = "[Not properly set]"
        
        print(f"Google OAuth Configuration:")
        print(f"Client ID: {client_id}")
        print(f"Client Secret: {masked_secret}")
        
        # Make sure to generate a full, absolute URL for the callback
        redirect_uri = url_for('google_authorize', _external=True)
        print(f"Redirecting to Google with callback URL: {redirect_uri}")
        return google.authorize_redirect(redirect_uri)
    except Exception as e:
        print(f"Error in google_login: {str(e)}")
        flash(f"Error initiating Google login: {str(e)}", "danger")
        return redirect(url_for('login'))


@app.route('/login/google/callback')
def google_authorize():
    try:
        # Print request details for debugging
        print("Received Google callback with params:")
        for key, value in request.args.items():
            print(f"  {key}: {value}")
        
        # Try to get the token without ID token validation
        token = None
        try:
            # First attempt: Try with explicit disabling of validation if supported
            token = google.authorize_access_token(id_token_validation=False)
        except TypeError:
            # Fallback: Just get the token and handle validation issues separately
            token = google.authorize_access_token()
        
        print(f"Received token: {token}")
        
        # Extract user info directly from the token if available
        user_info = None
        if token and 'userinfo' in token:
            user_info = token['userinfo']
            print(f"User info from token: {user_info}")
        
        # If user_info not in token, get it from Google API
        if not user_info:
            # Get user info from Google - Using full URL instead of relative path
            resp = google.get('https://www.googleapis.com/oauth2/v1/userinfo', token=token)
            user_info = resp.json()
            print(f"Google user info from API: {user_info}")
        
        # Get user ID (could be 'id' or 'sub' depending on response format)
        user_id = user_info.get('id') or user_info.get('sub')
        if not user_id:
            raise ValueError("Could not get user ID from Google response")
        
        # Get user email
        user_email = user_info.get('email')
        if not user_email:
            raise ValueError("Could not get email from Google response")
        
        # Check if user exists
        user = User.find_by_provider_id('google', user_id)
        
        if not user:
            # Check if email already exists
            email_user = User.find_by_email(user_email)
            if email_user:
                # Link accounts
                email_user.auth_provider = 'google'
                email_user.provider_id = user_id
                email_user.profile_picture = user_info.get('picture')
                email_user.update_last_login()
                db.session.commit()
                user = email_user
            else:
                # Create new user
                import secrets
                username = f"google_{user_id}"
                # Check if username exists and generate a unique one if needed
                while User.query.filter_by(username=username).first():
                    username = f"google_{user_id}_{secrets.token_hex(4)}"
                    
                user = User(
                    email=user_email,
                    username=username,
                    full_name=user_info.get('name'),
                    profile_picture=user_info.get('picture'),
                    auth_provider='google',
                    provider_id=user_id
                )
                db.session.add(user)
                db.session.commit()
        
        # Log in the user
        login_user(user)
        user.update_last_login()
        flash('Successfully logged in with Google', 'success')
        return redirect(url_for('index'))
        
    except Exception as e:
        print(f"Error in google_authorize: {str(e)}")
        import traceback
        traceback.print_exc()
        flash(f"Error logging in with Google: {str(e)}", "danger")
        return redirect(url_for('login'))

# Simplified map backend that reuses existing location data
# Add these routes to your web_app.py

# Add these enhanced routes to your web_app.py to support the new map functionality

# Add these enhanced routes to your web_app.py to support the new map functionality

@app.route("/map")
def map_view():
    """Render the main map interface with optional filters"""
    # Get URL parameters for pre-filtering
    selected_places = request.args.get('selected', '')
    highlight_place = request.args.get('highlight', '')
    album_id = request.args.get('album', '')
    filter_places = request.args.get('filter', '')
    
    # Pass these to the template for JavaScript to use
    return render_template("map.html", 
                         selected_places=selected_places,
                         highlight_place=highlight_place,
                         album_id=album_id,
                         filter_places=filter_places)


# Add these improved routes to your web_app.py

# Replace the get_map_places_improved function in web_app.py with this fixed version:

@app.route("/api/map/places")
def get_map_places_improved():
    """Enhanced API endpoint to get places with better coordinate handling"""
    try:
        # Get filter parameters
        genre_filter = request.args.get('genre')
        cuisine_filter = request.args.get('cuisine')
        min_rating = request.args.get('min_rating', type=float)
        city_filter = request.args.get('city')
        
        # Get selection parameters
        selected_places = request.args.get('selected', '').split(',') if request.args.get('selected') else []
        filter_places = request.args.get('filter', '').split(',') if request.args.get('filter') else []
        album_id = request.args.get('album')
        highlight_place = request.args.get('highlight')
        
        print(f"🔍 Map places request with params: selected={selected_places}, highlight={highlight_place}, album={album_id}")
        
        # Get results based on user authentication
        if current_user.is_authenticated:
            results = Result.query.filter_by(user_id=current_user.id).all()
        else:
            results = Result.query.filter_by(user_id=None).all()
        
        places_data = []
        seen_places = set()  # To avoid duplicates based on name+city
        
        for result in results:
            activities = result.data.get('activities', [])
            for activity_index, activity in enumerate(activities):
                place_name = activity.get('place_name')
                if not place_name:
                    continue
                
                availability = activity.get('availability', {})
                
                # Try multiple coordinate field names (your data might use different field names)
                latitude = None
                longitude = None
                
                # Check various possible field names for coordinates
                for lat_field in ['lat', 'latitude', 'y']:
                    if availability.get(lat_field):
                        latitude = availability[lat_field]
                        break
                        
                for lng_field in ['lon', 'lng', 'longitude', 'x']:
                    if availability.get(lng_field):
                        longitude = availability[lng_field]
                        break
                
                # Convert to float if they're strings
                try:
                    if latitude:
                        latitude = float(latitude)
                    if longitude:
                        longitude = float(longitude)
                except (ValueError, TypeError):
                    latitude = None
                    longitude = None
                
                # If we still don't have coordinates, try to geocode using the address
                if not latitude or not longitude:
                    street_address = availability.get('street_address', '')
                    city = availability.get('city', '')
                    
                    if street_address or city:
                        # Try to geocode the address
                        try:
                            import requests
                            api_key = os.getenv("GOOGLE_API_KEY")
                            if api_key:
                                address_query = street_address if street_address else f"{place_name}, {city}"
                                geocode_url = "https://maps.googleapis.com/maps/api/geocode/json"
                                params = {
                                    'address': address_query,
                                    'key': api_key
                                }
                                
                                response = requests.get(geocode_url, params=params, timeout=5)
                                if response.status_code == 200:
                                    data = response.json()
                                    if data['status'] == 'OK' and data['results']:
                                        location = data['results'][0]['geometry']['location']
                                        latitude = location['lat']
                                        longitude = location['lng']
                                        
                                        # Update the activity with the coordinates for future use
                                        activity['availability']['lat'] = latitude
                                        activity['availability']['lon'] = longitude
                                        
                                        # Save back to database
                                        result.data['activities'][activity_index] = activity
                                        flag_modified(result, "data")
                                        db.session.commit()
                                        
                                        print(f"✅ Geocoded {place_name}: {latitude}, {longitude}")
                                        
                        except Exception as e:
                            print(f"❌ Geocoding failed for {place_name}: {e}")
                
                # Skip if we still don't have coordinates
                if not latitude or not longitude:
                    print(f"⚠️ Skipping {place_name} - no coordinates available")
                    continue
                
                # Create activity reference
                activity_ref = f"{result.id}_{activity_index}"
                
                # Apply selection filters
                if selected_places and activity_ref not in selected_places:
                    # Also check the alternative format with dash
                    alt_ref = f"{result.id}-{activity_index}"
                    if alt_ref not in selected_places:
                        continue
                        
                if filter_places and activity_ref not in filter_places:
                    alt_ref = f"{result.id}-{activity_index}"
                    if alt_ref not in filter_places:
                        continue
                
                # Apply album filter
                if album_id:
                    album = db.session.get(Album, int(album_id))
                    if album:
                        album_refs = [ref for ref in album.activities]
                        if f"{result.id}-{activity_index}" not in album_refs:
                            continue
                
                # FIXED: Apply other filters with null checks
                if genre_filter and (not activity.get('genre') or activity.get('genre', '').lower() != genre_filter.lower()):
                    continue
                if cuisine_filter and (not activity.get('cuisine') or activity.get('cuisine', '').lower() != cuisine_filter.lower()):
                    continue
                if city_filter and (not availability.get('city') or availability.get('city', '').lower() != city_filter.lower()):
                    continue
                
                # FIXED: Create a unique identifier with proper null handling
                place_name_safe = place_name.lower() if place_name else "unknown"
                city_safe = availability.get('city', '') or ""
                city_safe = city_safe.lower() if city_safe else ""
                place_key = f"{place_name_safe}_{city_safe}"
                
                # Check if we've already processed this place
                if place_key in seen_places:
                    continue
                seen_places.add(place_key)
                
                # Get or generate image URL
                image_url = activity.get('image_url')
                if not image_url:
                    image_url = fetch_place_image(place_name, activity)
                
                # FIXED: Handle highlighting properly
                is_highlighted = False
                if highlight_place:
                    # Check multiple possible formats
                    possible_ids = [
                        f"{result.id}-{activity_index}",
                        f"{result.id}_{activity_index}",
                        str(result.id),
                        highlight_place
                    ]
                    is_highlighted = activity_ref in possible_ids or any(pid == highlight_place for pid in possible_ids)
                
                place_data = {
                    'id': activity_ref,
                    'name': place_name,
                    'genre': activity.get('genre', 'restaurant'),
                    'cuisine': activity.get('cuisine', ''),  # Can be None/empty
                    'street_address': availability.get('street_address', ''),
                    'city': availability.get('city', ''),
                    'state': availability.get('state', ''), 
                    'country': availability.get('country', ''),
                    'region': availability.get('region', ''),
                    'latitude': float(latitude),
                    'longitude': float(longitude),
                    'total_visits': 1,
                    'total_reviews': len(activity.get('user_reviews', [])),
                    'average_rating': 0.0,
                    'photo_url': image_url,
                    'result_id': result.id,
                    'activity_index': activity_index,
                    'source_url': result.url,
                    'visit_date': result.timestamp.isoformat(),
                    'vibes': activity.get('vibes', ''),
                    'visual_data': activity.get('visual_data', {}),
                    'is_selected': activity_ref in selected_places or f"{result.id}-{activity_index}" in selected_places,
                    'is_highlighted': is_highlighted,
                    # Add image_url for consistency
                    'image_url': image_url
                }
                
                # Calculate average rating from user reviews
                reviews = activity.get('user_reviews', [])
                if reviews:
                    total_rating = sum(review.get('rating', 0) for review in reviews)
                    place_data['average_rating'] = total_rating / len(reviews)
                
                # Apply rating filter
                if min_rating and place_data['average_rating'] < min_rating:
                    continue
                
                places_data.append(place_data)
        
        print(f"✅ Returning {len(places_data)} places")
        
        return jsonify({
            'places': places_data,
            'total': len(places_data),
            'filters_applied': {
                'genre': genre_filter,
                'cuisine': cuisine_filter,
                'city': city_filter,
                'min_rating': min_rating,
                'selected_count': len(selected_places) if selected_places else 0,
                'album_id': album_id,
                'highlight': highlight_place
            }
        })
        
    except Exception as e:
        app.logger.error(f"Error getting map places: {str(e)}")
        import traceback
        traceback.print_exc()
        return jsonify({'error': str(e), 'places': []}), 500


@app.route("/api/gallery/select")
def gallery_select():
    """API endpoint to get gallery items for selection"""
    try:
        if current_user.is_authenticated:
            results = Result.query.filter_by(user_id=current_user.id).order_by(Result.timestamp.desc()).all()
        else:
            results = Result.query.filter_by(user_id=None).order_by(Result.timestamp.desc()).all()
        
        gallery_items = []
        for result in results:
            activities = result.data.get('activities', [])
            for activity_index, activity in enumerate(activities):
                if activity.get('place_name'):
                    availability = activity.get('availability', {})
                    
                    # Check if has coordinates
                    has_location = bool(
                        availability.get('lat') or availability.get('latitude') or 
                        availability.get('street_address') or availability.get('city')
                    )
                    
                    gallery_items.append({
                        'id': f"{result.id}_{activity_index}",
                        'result_id': result.id,
                        'activity_index': activity_index,
                        'place_name': activity.get('place_name'),
                        'genre': activity.get('genre', 'restaurant'),
                        'cuisine': activity.get('cuisine', ''),
                        'city': availability.get('city', ''),
                        'image_url': activity.get('image_url', ''),
                        'visit_date': result.timestamp.isoformat(),
                        'source_url': result.url,
                        'has_location': has_location,
                        'vibes': activity.get('vibes', ''),
                        'food_items': [item.get('name', '') for item in activity.get('visual_data', {}).get('food_items', [])][:3]
                    })
        
        return jsonify({
            'items': gallery_items,
            'total': len(gallery_items)
        })
        
    except Exception as e:
        app.logger.error(f"Error getting gallery items: {str(e)}")
        return jsonify({'error': str(e), 'items': []}), 500





@app.route("/api/map/add-selected", methods=["POST"])
def add_selected_to_map():
    """Add selected gallery items to map view"""
    try:
        data = request.get_json()
        selected_ids = data.get('selected_ids', [])
        
        if not selected_ids:
            return jsonify({'error': 'No items selected'}), 400
        
        # Redirect URL with selected items
        selected_param = ','.join(selected_ids)
        redirect_url = f"/map?selected={selected_param}"
        
        return jsonify({
            'success': True,
            'redirect_url': redirect_url,
            'selected_count': len(selected_ids)
        })
        
    except Exception as e:
        app.logger.error(f"Error adding selected to map: {str(e)}")
        return jsonify({'error': str(e)}), 500

@app.route("/api/map/place/<result_id>/<int:activity_index>")
def get_place_details(result_id, activity_index):
    """Enhanced place details with more comprehensive information"""
    try:
        result = db.session.get(Result, int(result_id))
        if not result:
            return jsonify({'error': 'Result not found'}), 404
            
        # Check permissions
        if result.user_id and (not current_user.is_authenticated or result.user_id != current_user.id):
            return jsonify({'error': 'Permission denied'}), 403
        
        activities = result.data.get('activities', [])
        if activity_index >= len(activities):
            return jsonify({'error': 'Activity not found'}), 404
            
        activity = activities[activity_index]
        
        # Find other visits to the same place
        place_name = activity.get('place_name', '').lower()
        same_place_visits = []
        
        # Search through all user's results for the same place
        if current_user.is_authenticated:
            user_results = Result.query.filter_by(user_id=current_user.id).all()
        else:
            user_results = Result.query.filter_by(user_id=None).all()
            
        for r in user_results:
            for i, a in enumerate(r.data.get('activities', [])):
                if (a.get('place_name', '').lower() == place_name and 
                    not (r.id == result.id and i == activity_index)):  # Exclude current
                    same_place_visits.append({
                        'visit_date': r.timestamp.isoformat(),
                        'source_url': r.url,
                        'result_id': r.id,
                        'activity_index': i,
                        'dishes': a.get('dishes', []),
                        'vibes': a.get('vibes'),
                        'visual_data': a.get('visual_data', {}),
                        'user_reviews': a.get('user_reviews', [])
                    })
        
        # Sort visits by date
        same_place_visits.sort(key=lambda x: x['visit_date'], reverse=True)
        
        # Aggregate food items from all visits
        all_food_items = []
        if 'visual_data' in activity and 'food_items' in activity['visual_data']:
            all_food_items.extend(activity['visual_data']['food_items'])
            
        for visit in same_place_visits:
            if 'food_items' in visit.get('visual_data', {}):
                all_food_items.extend(visit['visual_data']['food_items'])
        
        # Remove duplicates
        unique_food_items = []
        seen_food = set()
        for item in all_food_items:
            name = item.get('name', '').lower()
            if name and name not in seen_food:
                unique_food_items.append(item)
                seen_food.add(name)
        
        # Aggregate all user reviews
        all_reviews = activity.get('user_reviews', [])
        for visit in same_place_visits:
            all_reviews.extend(visit.get('user_reviews', []))
        
        # Calculate overall rating
        overall_rating = 0.0
        if all_reviews:
            total_rating = sum(review.get('rating', 0) for review in all_reviews)
            overall_rating = total_rating / len(all_reviews)
        
        return jsonify({
            'place': {
                'name': activity.get('place_name'),
                'genre': activity.get('genre'),
                'cuisine': activity.get('cuisine'),
                'availability': activity.get('availability', {}),
                'vibes': activity.get('vibes'),
                'dishes': activity.get('dishes', []),
                'visual_data': activity.get('visual_data', {}),
                'photo_url': activity.get('image_url'),
                'total_visits': len(same_place_visits) + 1,
                'all_food_items': unique_food_items,
                'user_reviews': all_reviews,
                'overall_rating': overall_rating,
                'review_count': len(all_reviews)
            },
            'current_visit': {
                'visit_date': result.timestamp.isoformat(),
                'source_url': result.url,
                'result_id': result.id,
                'activity_index': activity_index
            },
            'other_visits': same_place_visits[:10],  # Limit to 10 recent visits
            'similar_places': []  # Could be enhanced with similarity search
        })
        
    except Exception as e:
        app.logger.error(f"Error getting place details: {str(e)}")
        return jsonify({'error': str(e)}), 500


@app.route("/api/map/similar/<result_id>/<int:activity_index>")
def find_similar_places(result_id, activity_index):
    """Find places similar to the given place"""
    try:
        result = db.session.get(Result, int(result_id))
        if not result:
            return jsonify({'error': 'Result not found'}), 404
            
        activities = result.data.get('activities', [])
        if activity_index >= len(activities):
            return jsonify({'error': 'Activity not found'}), 404
            
        target_activity = activities[activity_index]
        target_genre = target_activity.get('genre', '')
        target_cuisine = target_activity.get('cuisine', '')
        target_city = target_activity.get('availability', {}).get('city', '')
        
        # Find similar places
        if current_user.is_authenticated:
            user_results = Result.query.filter_by(user_id=current_user.id).all()
        else:
            user_results = Result.query.filter_by(user_id=None).all()
        
        similar_places = []
        
        for r in user_results:
            for i, a in enumerate(r.data.get('activities', [])):
                # Skip the same place
                if r.id == result.id and i == activity_index:
                    continue
                
                # Skip places without coordinates
                if not a.get('availability', {}).get('lat'):
                    continue
                
                similarity_score = 0
                
                # Genre similarity (high weight)
                if a.get('genre', '') == target_genre:
                    similarity_score += 40
                
                # Cuisine similarity (high weight for food places)
                if target_cuisine and a.get('cuisine', '') == target_cuisine:
                    similarity_score += 30
                
                # City similarity (medium weight)
                if a.get('availability', {}).get('city', '') == target_city:
                    similarity_score += 20
                
                # Visual similarity (based on detected objects/scenes)
                target_visual = target_activity.get('visual_data', {})
                activity_visual = a.get('visual_data', {})
                
                # Compare detected objects
                target_objects = set(obj.get('label', '').lower() for obj in target_visual.get('detected_objects', []))
                activity_objects = set(obj.get('label', '').lower() for obj in activity_visual.get('detected_objects', []))
                object_overlap = len(target_objects.intersection(activity_objects))
                similarity_score += object_overlap * 5
                
                # Compare food items
                target_foods = set(food.get('name', '').lower() for food in target_visual.get('food_items', []))
                activity_foods = set(food.get('name', '').lower() for food in activity_visual.get('food_items', []))
                food_overlap = len(target_foods.intersection(activity_foods))
                similarity_score += food_overlap * 10
                
                # Only include places with reasonable similarity
                if similarity_score >= 20:
                    similar_places.append({
                        'result_id': r.id,
                        'activity_index': i,
                        'place_name': a.get('place_name'),
                        'genre': a.get('genre'),
                        'cuisine': a.get('cuisine'),
                        'city': a.get('availability', {}).get('city'),
                        'image_url': a.get('image_url'),
                        'similarity_score': similarity_score,
                        'latitude': a.get('availability', {}).get('lat'),
                        'longitude': a.get('availability', {}).get('lon')
                    })
        
        # Sort by similarity score
        similar_places.sort(key=lambda x: x['similarity_score'], reverse=True)
        
        return jsonify({
            'target_place': {
                'name': target_activity.get('place_name'),
                'genre': target_genre,
                'cuisine': target_cuisine,
                'city': target_city
            },
            'similar_places': similar_places[:10]  # Return top 10
        })
        
    except Exception as e:
        app.logger.error(f"Error finding similar places: {str(e)}")
        return jsonify({'error': str(e)}), 500


@app.route("/api/map/export/<format>")
def export_map_data(format):
    """Export map data in various formats (json, csv, kml)"""
    try:
        # Get places data (reuse the existing logic)
        places_response = get_map_places()
        places_data = places_response.get_json()['places']
        
        if format.lower() == 'json':
            return jsonify(places_data)
            
        elif format.lower() == 'csv':
            import csv
            import io
            
            output = io.StringIO()
            writer = csv.DictWriter(output, fieldnames=[
                'name', 'genre', 'cuisine', 'city', 'country', 
                'latitude', 'longitude', 'street_address', 'vibes', 
                'total_visits', 'average_rating', 'source_url'
            ])
            
            writer.writeheader()
            for place in places_data:
                writer.writerow({
                    'name': place['name'],
                    'genre': place['genre'],
                    'cuisine': place.get('cuisine', ''),
                    'city': place.get('city', ''),
                    'country': place.get('country', ''),
                    'latitude': place['latitude'],
                    'longitude': place['longitude'],
                    'street_address': place.get('street_address', ''),
                    'vibes': place.get('vibes', ''),
                    'total_visits': place['total_visits'],
                    'average_rating': place['average_rating'],
                    'source_url': place['source_url']
                })
            
            output.seek(0)
            return output.getvalue(), 200, {
                'Content-Type': 'text/csv',
                'Content-Disposition': 'attachment; filename=places_export.csv'
            }
            
        elif format.lower() == 'kml':
            # Generate KML for Google Earth
            kml_content = generate_kml(places_data)
            return kml_content, 200, {
                'Content-Type': 'application/vnd.google-earth.kml+xml',
                'Content-Disposition': 'attachment; filename=places_export.kml'
            }
            
        else:
            return jsonify({'error': 'Unsupported format'}), 400
            
    except Exception as e:
        app.logger.error(f"Error exporting map data: {str(e)}")
        return jsonify({'error': str(e)}), 500


def generate_kml(places_data):
    """Generate KML content for Google Earth export"""
    kml_header = '''<?xml version="1.0" encoding="UTF-8"?>
<kml xmlns="http://www.opengis.net/kml/2.2">
<Document>
    <name>Places from Video Analysis</name>
    <description>Places discovered through video analysis</description>
'''

    kml_footer = '''</Document>
</kml>'''

    placemarks = []
    for place in places_data:
        placemark = f'''
    <Placemark>
        <name>{place['name']}</name>
        <description>
            <![CDATA[
                <b>Type:</b> {place['genre']}<br/>
                {f"<b>Cuisine:</b> {place.get('cuisine', '')}<br/>" if place.get('cuisine') else ""}
                <b>Location:</b> {place.get('city', '')}, {place.get('country', '')}<br/>
                <b>Visits:</b> {place['total_visits']}<br/>
                {f"<b>Rating:</b> {place['average_rating']:.1f}/5<br/>" if place['average_rating'] > 0 else ""}
                <b>Source:</b> <a href="{place['source_url']}">View Original Video</a>
            ]]>
        </description>
        <Point>
            <coordinates>{place['longitude']},{place['latitude']},0</coordinates>
        </Point>
    </Placemark>'''
        placemarks.append(placemark)

    return kml_header + ''.join(placemarks) + kml_footer


@app.route("/api/map/cluster")
def get_place_clusters():
    """Get clustered places for better map performance with many markers"""
    try:
        zoom_level = request.args.get('zoom', 10, type=int)
        
        # Get all places
        places_response = get_map_places()
        places_data = places_response.get_json()['places']
        
        if zoom_level < 8:
            # At low zoom, cluster by city
            clusters = {}
            for place in places_data:
                city = place.get('city', 'Unknown')
                if city not in clusters:
                    clusters[city] = {
                        'name': city,
                        'count': 0,
                        'latitude': 0,
                        'longitude': 0,
                        'places': []
                    }
                
                clusters[city]['count'] += 1
                clusters[city]['latitude'] += place['latitude']
                clusters[city]['longitude'] += place['longitude']
                clusters[city]['places'].append(place)
            
            # Calculate average coordinates for each cluster
            for cluster in clusters.values():
                if cluster['count'] > 0:
                    cluster['latitude'] /= cluster['count']
                    cluster['longitude'] /= cluster['count']
            
            return jsonify({
                'clusters': list(clusters.values()),
                'zoom_level': zoom_level,
                'type': 'city_clusters'
            })
        else:
            # At higher zoom, return individual places
            return jsonify({
                'clusters': [{'name': p['name'], 'count': 1, 'latitude': p['latitude'], 
                             'longitude': p['longitude'], 'places': [p]} for p in places_data],
                'zoom_level': zoom_level,
                'type': 'individual_places'
            })
            
    except Exception as e:
        app.logger.error(f"Error getting place clusters: {str(e)}")
        return jsonify({'error': str(e)}), 500
    


    
def format_full_address(availability):
    """Format a complete address from availability data"""
    address_parts = []
    
    street = availability.get('street_address', '')
    if street:
        address_parts.append(street)
    
    city = availability.get('city', '')
    state = availability.get('state', '')
    country = availability.get('country', '')
    
    city_state_country = []
    if city:
        city_state_country.append(city)
    if state:
        city_state_country.append(state)
    if country:
        city_state_country.append(country)
    
    if city_state_country:
        address_parts.append(', '.join(city_state_country))
    
    return ', '.join(address_parts) if address_parts else 'Address not available'

@app.route("/api/map/address-suggestions")
def get_address_suggestions():
    """Get address-based search suggestions"""
    try:
        query = request.args.get('q', '').strip()
        scope = request.args.get('scope', 'mine')
        
        if len(query) < 2:
            return jsonify({'suggestions': []})
        
        # Build results query based on scope
        results_query = Result.query
        if scope == 'mine' and current_user.is_authenticated:
            results_query = results_query.filter_by(user_id=current_user.id)
        elif scope == 'friends' and current_user.is_authenticated:
            friends = current_user.get_friends()
            friend_ids = [f.id for f in friends]
            results_query = results_query.filter(Result.user_id.in_(friend_ids))
        
        results = results_query.all()
        
        suggestions = set()
        
        for result in results:
            activities = result.data.get('activities', [])
            for activity in activities:
                availability = activity.get('availability', {})
                
                # Collect address-related suggestions
                street_address = availability.get('street_address', '')
                city = availability.get('city', '')
                state = availability.get('state', '')
                country = availability.get('country', '')
                place_name = activity.get('place_name', '')
                
                # Add suggestions that match the query
                potential_suggestions = [
                    street_address,
                    city,
                    state,
                    country,
                    place_name,
                    f"{place_name}, {city}" if place_name and city else "",
                    f"{city}, {state}" if city and state else "",
                    f"{city}, {country}" if city and country else ""
                ]
                
                for suggestion in potential_suggestions:
                    if suggestion and len(suggestion) > 2 and query.lower() in suggestion.lower():
                        suggestions.add(suggestion)
                        
                        # Limit suggestions
                        if len(suggestions) >= 10:
                            break
                
                if len(suggestions) >= 10:
                    break
        
        return jsonify({
            'suggestions': sorted(list(suggestions))[:10],
            'type': 'address_based'
        })
        
    except Exception as e:
        app.logger.error(f"Error getting address suggestions: {str(e)}")
        return jsonify({'error': str(e), 'suggestions': []}), 500


def process_activity_for_map(db_result, activity_index, activity, lat, lng, radius, scope, min_rating):
    """Process a single activity for map display"""
    place_name = activity.get('place_name', '')
    if not place_name:
        return None
    
    availability = activity.get('availability', {})
    place_lat = availability.get('lat')
    place_lng = availability.get('lon')
    
    # Skip places without coordinates for nearby search
    if scope == 'nearby' and (not place_lat or not place_lng):
        return None
    
    # Calculate distance for nearby search
    distance_km = None
    if scope == 'nearby' and lat and lng and place_lat and place_lng:
        distance_km = calculate_distance(lat, lng, place_lat, place_lng)
        if distance_km > radius:
            return None
    
    # Calculate rating
    reviews = activity.get('user_reviews', [])
    avg_rating = 0.0
    if reviews:
        avg_rating = sum(r.get('rating', 0) for r in reviews) / len(reviews)
    
    # Apply rating filter
    if min_rating and avg_rating < min_rating:
        return None
    
    # Get user info for the place
    place_user = None
    if db_result.user_id:
        place_user = db.session.get(User, db_result.user_id)
    
    return {
        'id': f"{db_result.id}_{activity_index}",
        'result_id': db_result.id,
        'activity_index': activity_index,
        'name': place_name,
        'genre': activity.get('genre', ''),
        'cuisine': activity.get('cuisine', ''),
        'street_address': availability.get('street_address', ''),
        'city': availability.get('city', ''),
        'state': availability.get('state', ''),
        'country': availability.get('country', ''),
        'latitude': place_lat,
        'longitude': place_lng,
        'distance_km': distance_km,
        'rating': avg_rating,
        'review_count': len(reviews),
        'vibes': activity.get('vibes', ''),
        'image_url': activity.get('image_url', ''),
        'source_url': db_result.url,
        'visit_date': db_result.timestamp.isoformat(),
        'visual_data': activity.get('visual_data', {}),
        'user': {
            'id': db_result.user_id,
            'username': place_user.username if place_user else 'Anonymous',
            'profile_picture': place_user.profile_picture if place_user else None,
            'is_friend': current_user.is_friend_with(db_result.user_id) if current_user.is_authenticated and db_result.user_id else False
        } if db_result.user_id else None,
        'scope': scope
    }


def perform_keyword_search(query, scope, lat, lng, radius, genre, cuisine, min_rating):
    """Fallback keyword search when semantic search isn't available"""
    # Build base query based on scope
    results_query = Result.query
    
    if scope == 'mine':
        if current_user.is_authenticated:
            results_query = results_query.filter_by(user_id=current_user.id)
        else:
            results_query = results_query.filter_by(user_id=None)
            
    elif scope == 'friends':
        if current_user.is_authenticated:
            friends = current_user.get_friends()
            friend_ids = [f.id for f in friends]
            results_query = results_query.filter(Result.user_id.in_(friend_ids))
        else:
            return []
            
    elif scope == 'global':
        # All public results
        pass
        
    elif scope == 'nearby':
        if not lat or not lng:
            return []
    
    results = results_query.all()
    
    # Process all activities and apply filters
    places = []
    for result in results:
        activities = result.data.get('activities', [])
        for activity_index, activity in enumerate(activities):
            # Apply text search filter if query provided
            if query:
                place_name = activity.get('place_name', '')
                searchable_text = ' '.join([
                    place_name.lower(),
                    activity.get('genre', '').lower(),
                    activity.get('cuisine', '').lower(),
                    activity.get('vibes', '').lower(),
                    activity.get('availability', {}).get('city', '').lower(),
                    activity.get('availability', {}).get('country', '').lower(),
                ])
                
                # Also search in food items
                food_items = activity.get('visual_data', {}).get('food_items', [])
                food_text = ' '.join([item.get('name', '').lower() for item in food_items])
                searchable_text += ' ' + food_text
                
                if query.lower() not in searchable_text:
                    continue
            
            # Apply genre filter
            if genre and activity.get('genre', '').lower() != genre.lower():
                continue
            
            # Apply cuisine filter
            if cuisine and activity.get('cuisine', '').lower() != cuisine.lower():
                continue
            
            place_data = process_activity_for_map(
                result, activity_index, activity, 
                lat, lng, radius, scope, min_rating
            )
            if place_data:
                places.append(place_data)
    
    return places


def apply_final_filters_and_sorting(places, request_args):
    """Apply final sorting to search results"""
    sort_by = request_args.get('sort', 'relevance')
    
    if sort_by == 'distance':
        places.sort(key=lambda x: x.get('distance_km') or 999)
    elif sort_by == 'rating':
        places.sort(key=lambda x: x.get('rating', 0), reverse=True)
    elif sort_by == 'recent':
        places.sort(key=lambda x: x.get('visit_date', ''), reverse=True)
    elif sort_by == 'name':
        places.sort(key=lambda x: x.get('name', '').lower())
    elif sort_by == 'relevance':
        # If we have relevance scores from semantic search, sort by those
        # Otherwise sort by a combination of factors
        if any(p.get('relevance_score') for p in places):
            places.sort(key=lambda x: x.get('relevance_score', 0), reverse=True)
        else:
            # Sort by rating as a proxy for relevance
            places.sort(key=lambda x: x.get('rating', 0), reverse=True)
    
    return places


@app.route("/api/map/semantic-suggestions")
def get_semantic_suggestions():
    """Get semantic search suggestions using vector similarity"""
    try:
        query = request.args.get('q', '').strip()
        scope = request.args.get('scope', 'mine')
        
        if len(query) < 2:
            return jsonify({'suggestions': []})
        
        # Use vector search to find similar content
        try:
            # Get semantic results for suggestions
            semantic_results = vector_manager.search(
                query=query,
                limit=20,  # Get more for variety
                filters={}  # No filters for suggestions
            )
            
            suggestions = set()
            
            for result in semantic_results[:10]:  # Use top 10 for suggestions
                result_id = result.get('id')
                if not result_id:
                    continue
                    
                db_result = db.session.get(Result, int(result_id))
                if not db_result:
                    continue
                
                activities = db_result.data.get('activities', [])
                for activity in activities:
                    # Add place names
                    place_name = activity.get('place_name', '')
                    if place_name and len(place_name) > 2:
                        suggestions.add(place_name)
                    
                    # Add cuisines
                    cuisine = activity.get('cuisine', '')
                    if cuisine and len(cuisine) > 2:
                        suggestions.add(cuisine)
                    
                    # Add genres
                    genre = activity.get('genre', '')
                    if genre and len(genre) > 2:
                        suggestions.add(genre)
                    
                    # Add food items
                    food_items = activity.get('visual_data', {}).get('food_items', [])
                    for item in food_items:
                        food_name = item.get('name', '')
                        if food_name and len(food_name) > 2:
                            suggestions.add(food_name)
                    
                    # Limit suggestions
                    if len(suggestions) >= 15:
                        break
                
                if len(suggestions) >= 15:
                    break
            
            # Sort suggestions by relevance to query
            sorted_suggestions = sorted(list(suggestions))[:10]
            
            return jsonify({'suggestions': sorted_suggestions, 'type': 'semantic'})
            
        except Exception as e:
            print(f"Semantic suggestions failed: {e}")
            # Fall back to keyword-based suggestions
            return get_keyword_suggestions(query, scope)
            
    except Exception as e:
        app.logger.error(f"Error getting semantic suggestions: {str(e)}")
        return jsonify({'error': str(e), 'suggestions': []}), 500


def get_keyword_suggestions(query, scope):
    """Fallback keyword-based suggestions"""
    # Build results query based on scope
    results_query = Result.query
    if scope == 'mine' and current_user.is_authenticated:
        results_query = results_query.filter_by(user_id=current_user.id)
    elif scope == 'friends' and current_user.is_authenticated:
        friends = current_user.get_friends()
        friend_ids = [f.id for f in friends]
        results_query = results_query.filter(Result.user_id.in_(friend_ids))
    
    results = results_query.all()
    
    suggestions = set()
    
    for result in results:
        activities = result.data.get('activities', [])
        for activity in activities:
            # Place names
            place_name = activity.get('place_name', '')
            if query.lower() in place_name.lower():
                suggestions.add(place_name)
            
            # Cuisines
            cuisine = activity.get('cuisine', '')
            if cuisine and query.lower() in cuisine.lower():
                suggestions.add(cuisine)
            
            # Cities
            city = activity.get('availability', {}).get('city', '')
            if city and query.lower() in city.lower():
                suggestions.add(city)
            
            # Food items
            food_items = activity.get('visual_data', {}).get('food_items', [])
            for item in food_items:
                food_name = item.get('name', '')
                if food_name and query.lower() in food_name.lower():
                    suggestions.add(food_name)
    
    # Sort and limit suggestions
    sorted_suggestions = sorted(list(suggestions))[:10]
    
    return jsonify({'suggestions': sorted_suggestions, 'type': 'keyword'})


@app.route("/api/map/similar-places/<result_id>/<int:activity_index>")
def find_similar_places_semantic(result_id, activity_index):
    """Find places similar to the given place using semantic search"""
    try:
        result = db.session.get(Result, int(result_id))
        if not result:
            return jsonify({'error': 'Result not found'}), 404
            
        activities = result.data.get('activities', [])
        if activity_index >= len(activities):
            return jsonify({'error': 'Activity not found'}), 404
            
        target_activity = activities[activity_index]
        
        # Build a query from the target activity
        query_parts = []
        if target_activity.get('place_name'):
            query_parts.append(target_activity['place_name'])
        if target_activity.get('genre'):
            query_parts.append(target_activity['genre'])
        if target_activity.get('cuisine'):
            query_parts.append(target_activity['cuisine'])
        
        # Add food items for better matching
        food_items = target_activity.get('visual_data', {}).get('food_items', [])
        for item in food_items[:3]:  # Top 3 food items
            if item.get('name'):
                query_parts.append(item['name'])
        
        search_query = ' '.join(query_parts)
        
        try:
            # Use semantic search to find similar places
            similar_results = vector_manager.search(
                query=search_query,
                limit=20,
                filters={}
            )
            
            similar_places = []
            
            for sim_result in similar_results:
                sim_result_id = sim_result.get('id')
                if not sim_result_id or sim_result_id == result_id:
                    continue  # Skip the same result
                    
                db_sim_result = db.session.get(Result, int(sim_result_id))
                if not db_sim_result:
                    continue
                
                sim_activities = db_sim_result.data.get('activities', [])
                for i, sim_activity in enumerate(sim_activities):
                    # Skip places without coordinates
                    if not sim_activity.get('availability', {}).get('lat'):
                        continue
                    
                    similarity_score = sim_result.get('score', 0)
                    
                    similar_places.append({
                        'result_id': db_sim_result.id,
                        'activity_index': i,
                        'place_name': sim_activity.get('place_name'),
                        'genre': sim_activity.get('genre'),
                        'cuisine': sim_activity.get('cuisine'),
                        'city': sim_activity.get('availability', {}).get('city'),
                        'image_url': sim_activity.get('image_url'),
                        'similarity_score': similarity_score,
                        'latitude': sim_activity.get('availability', {}).get('lat'),
                        'longitude': sim_activity.get('availability', {}).get('lon')
                    })
            
            # Sort by similarity score
            similar_places.sort(key=lambda x: x['similarity_score'], reverse=True)
            
            return jsonify({
                'target_place': {
                    'name': target_activity.get('place_name'),
                    'genre': target_activity.get('genre'),
                    'cuisine': target_activity.get('cuisine'),
                    'city': target_activity.get('availability', {}).get('city')
                },
                'similar_places': similar_places[:10],  # Return top 10
                'search_type': 'semantic'
            })
            
        except Exception as e:
            print(f"Semantic similarity search failed: {e}")
            # Fall back to the original similarity algorithm
            return find_similar_places_keyword(result_id, activity_index)
            
    except Exception as e:
        app.logger.error(f"Error finding similar places: {str(e)}")
        return jsonify({'error': str(e)}), 500


def find_similar_places_keyword(result_id, activity_index):
    """Fallback keyword-based similarity search"""
    # Your existing similarity logic from the original routes
    # This is the same as the original find_similar_places function
    result = db.session.get(Result, int(result_id))
    activities = result.data.get('activities', [])
    target_activity = activities[activity_index]
    
    target_genre = target_activity.get('genre', '')
    target_cuisine = target_activity.get('cuisine', '')
    target_city = target_activity.get('availability', {}).get('city', '')
    
    # Find similar places using keyword matching
    if current_user.is_authenticated:
        user_results = Result.query.filter_by(user_id=current_user.id).all()
    else:
        user_results = Result.query.filter_by(user_id=None).all()
    
    similar_places = []
    
    for r in user_results:
        for i, a in enumerate(r.data.get('activities', [])):
            # Skip the same place
            if r.id == result.id and i == activity_index:
                continue
            
            # Skip places without coordinates
            if not a.get('availability', {}).get('lat'):
                continue
            
            similarity_score = 0
            
            # Genre similarity (high weight)
            if a.get('genre', '') == target_genre:
                similarity_score += 40
            
            # Cuisine similarity (high weight for food places)
            if target_cuisine and a.get('cuisine', '') == target_cuisine:
                similarity_score += 30
            
            # City similarity (medium weight)
            if a.get('availability', {}).get('city', '') == target_city:
                similarity_score += 20
            
            # Visual similarity (based on detected objects/scenes)
            target_visual = target_activity.get('visual_data', {})
            activity_visual = a.get('visual_data', {})
            
            # Compare detected objects
            target_objects = set(obj.get('label', '').lower() for obj in target_visual.get('detected_objects', []))
            activity_objects = set(obj.get('label', '').lower() for obj in activity_visual.get('detected_objects', []))
            object_overlap = len(target_objects.intersection(activity_objects))
            similarity_score += object_overlap * 5
            
            # Compare food items
            target_foods = set(food.get('name', '').lower() for food in target_visual.get('food_items', []))
            activity_foods = set(food.get('name', '').lower() for food in activity_visual.get('food_items', []))
            food_overlap = len(target_foods.intersection(activity_foods))
            similarity_score += food_overlap * 10
            
            # Only include places with reasonable similarity
            if similarity_score >= 20:
                similar_places.append({
                    'result_id': r.id,
                    'activity_index': i,
                    'place_name': a.get('place_name'),
                    'genre': a.get('genre'),
                    'cuisine': a.get('cuisine'),
                    'city': a.get('availability', {}).get('city'),
                    'image_url': a.get('image_url'),
                    'similarity_score': similarity_score,
                    'latitude': a.get('availability', {}).get('lat'),
                    'longitude': a.get('availability', {}).get('lon')
                })
    
    # Sort by similarity score
    similar_places.sort(key=lambda x: x['similarity_score'], reverse=True)
    
    return jsonify({
        'target_place': {
            'name': target_activity.get('place_name'),
            'genre': target_genre,
            'cuisine': target_cuisine,
            'city': target_city
        },
        'similar_places': similar_places[:10],  # Return top 10
        'search_type': 'keyword'
    })




# Instagram OAuth routes
@app.route('/login/instagram')
def instagram_login():
    redirect_uri = url_for('instagram_authorize', _external=True)
    return instagram.authorize_redirect(redirect_uri)


@app.route('/login/instagram/callback')
def instagram_authorize():
    try:
        token_response = instagram.authorize_access_token()
        
        # Get user info
        user_id = token_response.get('user_id')
        access_token = token_response.get('access_token')
        
        # Get detailed user info from Instagram Graph API
        response = requests.get(
            f"https://graph.instagram.com/me?fields=id,username&access_token={access_token}"
        )
        user_info = response.json()
        
        # Check if user exists
        user = User.find_by_provider_id('instagram', user_id)
        
        if not user:
            # Create new user (note: Instagram OAuth doesn't provide email by default)
            import secrets
            username = f"instagram_{user_info.get('username', user_id)}"
            
            # Check if username exists and generate a unique one if needed
            while User.query.filter_by(username=username).first():
                username = f"instagram_{user_id}_{secrets.token_hex(4)}"
                
            user = User(
                email=f"{username}@instagram.placeholder",  # Placeholder email
                username=username,
                full_name=user_info.get('username'),
                auth_provider='instagram',
                provider_id=user_id
            )
            db.session.add(user)
            db.session.commit()
        
        # Log in the user
        login_user(user)
        user.update_last_login()
        flash('Successfully logged in with Instagram', 'success')
        return redirect(url_for('index'))
        
    except Exception as e:
        flash(f'Error logging in with Instagram: {str(e)}', 'danger')
        return redirect(url_for('login'))


# Error handlers
@app.errorhandler(404)
def page_not_found(e):
    return render_template('errors/404.html'), 404


@app.errorhandler(500)
def internal_server_error(e):
    return render_template('errors/500.html'), 500


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5001, debug=False)