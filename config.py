"""Configuration settings for the application."""

import os
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

class Config:
    """Base configuration."""
    
    # Flask
    SECRET_KEY = os.getenv('SECRET_KEY', 'dev-key-change-in-production')
    
    # Database
    SQLALCHEMY_DATABASE_URI = os.getenv('DATABASE_URL', 'sqlite:///results.db')
    SQLALCHEMY_TRACK_MODIFICATIONS = False
    
    # API Keys
    GOOGLE_API_KEY = os.getenv('GOOGLE_API_KEY', '')
    OPENAI_API_KEY = os.getenv('OPENAI_API_KEY', '')
    
    # Pipeline settings
    BACKEND_MODE = os.getenv('BACKEND_MODE', 'local')
    ASR_URL = os.getenv('ASR_URL', 'http://127.0.0.1:8002/asr')
    VISION_URL = os.getenv('VISION_URL', 'http://127.0.0.1:8001/infer')
    S3_BUCKET = os.getenv('S3_BUCKET', 'video-analysis-dev')
    
    # OpenAI configuration
    OPENAI_MODEL = os.getenv('OPENAI_MODEL', 'gpt-4-vision-preview')
    
    # Feature flags
    ENABLE_FOOD_ANALYSIS = os.getenv('ENABLE_FOOD_ANALYSIS', 'true').lower() == 'true'
    
    # Performance settings
    MAX_FRAMES = int(os.getenv('MAX_FRAMES', '10'))
    CONFIDENCE_THRESHOLD = float(os.getenv('CONFIDENCE_THRESHOLD', '0.5'))


class DevelopmentConfig(Config):
    """Development configuration."""
    
    DEBUG = True
    TESTING = False


class TestingConfig(Config):
    """Testing configuration."""
    
    DEBUG = False
    TESTING = True
    SQLALCHEMY_DATABASE_URI = 'sqlite:///:memory:'


class ProductionConfig(Config):
    """Production configuration."""
    
    DEBUG = False
    TESTING = False

    # In production, ensure SECRET_KEY is set in environment
    SECRET_KEY = os.getenv('SECRET_KEY')
    if not SECRET_KEY:
        raise ValueError("No SECRET_KEY set for production environment")


# Determine which configuration to use based on environment
config = {
    'development': DevelopmentConfig,
    'testing': TestingConfig,
    'production': ProductionConfig,
    'default': DevelopmentConfig
}

def get_config():
    """Get the appropriate configuration based on environment."""
    env = os.getenv('FLASK_ENV', 'development')
    return config.get(env, config['default'])
