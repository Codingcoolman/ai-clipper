import os
from dotenv import load_dotenv

load_dotenv()

class Config:
    SECRET_KEY = os.getenv('SECRET_KEY', os.urandom(24).hex())
    MAX_CONTENT_LENGTH = 500 * 1024 * 1024  # 500MB max file size
    UPLOAD_FOLDER = os.path.abspath('output')
    
class ProductionConfig(Config):
    DEBUG = False
    TESTING = False
    # Add any production-specific settings here
    
class DevelopmentConfig(Config):
    DEBUG = True
    TESTING = True 