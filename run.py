import os
import sys
import logging
from flask_cors import CORS

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    stream=sys.stdout
)
logger = logging.getLogger(__name__)

# Add the current directory to PYTHONPATH
current_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.append(current_dir)
logger.info(f"Added {current_dir} to PYTHONPATH")

try:
    from src.app import app
    logger.info("Successfully imported Flask app")
except Exception as e:
    logger.error(f"Failed to import Flask app: {str(e)}")
    raise

# Enable CORS with proper configuration
try:
    CORS(app, 
         resources={r"/*": {
             "origins": ["https://init-12295.web.app"],  # Only allow your frontend
             "methods": ["GET", "POST", "PUT", "DELETE", "OPTIONS", "HEAD"],
             "allow_headers": ["Content-Type", "Authorization", "X-Requested-With", "Accept", "Origin"],
             "expose_headers": ["Content-Range", "X-Content-Range"],
             "supports_credentials": True,
             "max_age": 3600,
             "send_wildcard": False,
             "vary_header": True,
             "allow_private_network": True
         }},
         always_send=True,  # Always send CORS headers
         automatic_options=True,  # Handle OPTIONS automatically
         intercept_exceptions=True  # Add CORS headers to error responses
    )
    logger.info("CORS configuration applied successfully")
except Exception as e:
    logger.error(f"Failed to configure CORS: {str(e)}")
    raise

# Set absolute paths for template and static folders
try:
    app.template_folder = os.path.abspath('templates')
    app.static_folder = os.path.abspath('static')
    app.static_url_path = '/static'
    logger.info(f"Template folder set to: {app.template_folder}")
    logger.info(f"Static folder set to: {app.static_folder}")
except Exception as e:
    logger.error(f"Failed to set template/static folders: {str(e)}")
    raise

if __name__ == '__main__':
    try:
        port = int(os.environ.get('PORT', 10000))
        logger.info(f"Starting server on port {port}")
        app.run(host='0.0.0.0', port=port)
    except Exception as e:
        logger.error(f"Failed to start server: {str(e)}")
        raise 