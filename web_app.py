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


@app.route("/photo/<path:photo_ref>")
def google_photo(photo_ref: str):
    """
    Streams a Google Places photo to the browser so that the
    front‑end never sees your Maps API key.
    """
    api_key = os.getenv("GOOGLE_API_KEY")
    if not api_key:
        print("Error: GOOGLE_API_KEY not set")
        return "GOOGLE_API_KEY not set", 500

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
    try:
        # Try original query with user_id filter
        albums = Album.query.filter_by(user_id=None).order_by(Album.timestamp.desc()).all()
    except Exception as e:
        print(f"Warning: {e}")
        # Fall back to query without user_id filter
        albums = Album.query.order_by(Album.timestamp.desc()).all()
    
    return jsonify(
        [
            {
                "id": album.id,
                "name": album.name,
                "activities": album.activities,
                "timestamp": album.timestamp.isoformat() if album.timestamp else None,
            }
            for album in albums
        ]
    )


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