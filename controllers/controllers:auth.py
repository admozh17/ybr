from flask import Blueprint, request, render_template, redirect, url_for, flash, session, jsonify, current_app
from flask_login import login_user, logout_user, login_required, current_user
from models.auth import User
from app import db
import os
import json
from datetime import datetime
import secrets
from authlib.integrations.flask_client import OAuth
import requests

# Create auth blueprint
auth_bp = Blueprint('auth', __name__)

# Initialize OAuth
oauth = OAuth()

# Set up Google OAuth
google = oauth.register(
    name='google',
    client_id=os.getenv('GOOGLE_OAUTH_CLIENT_ID'),
    client_secret=os.getenv('GOOGLE_OAUTH_CLIENT_SECRET'),
    access_token_url='https://accounts.google.com/o/oauth2/token',
    access_token_params=None,
    authorize_url='https://accounts.google.com/o/oauth2/auth',
    authorize_params=None,
    api_base_url='https://www.googleapis.com/oauth2/v1/',
    client_kwargs={'scope': 'openid email profile'},
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

@auth_bp.route('/login', methods=['GET', 'POST'])
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

@auth_bp.route('/register', methods=['GET', 'POST'])
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
        return redirect(url_for('auth.login'))
        
    return render_template('auth/register.html')

@auth_bp.route('/logout')
@login_required
def logout():
    logout_user()
    flash('You have been logged out', 'info')
    return redirect(url_for('index'))

@auth_bp.route('/profile')
@login_required
def profile():
    return render_template('auth/profile.html', user=current_user)

@auth_bp.route('/profile/edit', methods=['GET', 'POST'])
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
        return redirect(url_for('auth.profile'))
        
    return render_template('auth/edit_profile.html')

# Google OAuth routes
@auth_bp.route('/login/google')
def google_login():
    redirect_uri = url_for('auth.google_authorize', _external=True)
    return google.authorize_redirect(redirect_uri)

@auth_bp.route('/login/google/callback')
def google_authorize():
    token = google.authorize_access_token()
    resp = google.get('userinfo')
    user_info = resp.json()
    
    # Check if user exists
    user = User.find_by_provider_id('google', user_info['id'])
    
    if not user:
        # Check if email already exists
        email_user = User.find_by_email(user_info['email'])
        if email_user:
            # Link accounts
            email_user.auth_provider = 'google'
            email_user.provider_id = user_info['id']
            email_user.profile_picture = user_info.get('picture')
            email_user.update_last_login()
            db.session.commit()
            user = email_user
        else:
            # Create new user
            username = f"google_{user_info['id']}"
            # Check if username exists and generate a unique one if needed
            while User.query.filter_by(username=username).first():
                username = f"google_{user_info['id']}_{secrets.token_hex(4)}"
                
            user = User(
                email=user_info['email'],
                username=username,
                full_name=user_info.get('name'),
                profile_picture=user_info.get('picture'),
                auth_provider='google',
                provider_id=user_info['id']
            )
            db.session.add(user)
            db.session.commit()
    
    # Log in the user
    login_user(user)
    user.update_last_login()
    flash('Successfully logged in with Google', 'success')
    return redirect(url_for('index'))

# Instagram OAuth routes
@auth_bp.route('/login/instagram')
def instagram_login():
    redirect_uri = url_for('auth.instagram_authorize', _external=True)
    return instagram.authorize_redirect(redirect_uri)

@auth_bp.route('/login/instagram/callback')
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
        return redirect(url_for('auth.login'))

# Initialize OAuth with the Flask app
def init_oauth(app):
    oauth.init_app(app)