import os
import sys
from flask_cors import CORS

# Add the current directory to PYTHONPATH
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from src.app import app

# Enable CORS with proper configuration
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

# Set absolute paths for template and static folders
app.template_folder = os.path.abspath('templates')
app.static_folder = os.path.abspath('static')
app.static_url_path = '/static'

if __name__ == '__main__':
    port = int(os.environ.get('PORT', 10000))
    app.run(host='0.0.0.0', port=port) 