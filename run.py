import os
import sys
from flask_cors import CORS

# Add the current directory to PYTHONPATH
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from src.app import app

# Enable CORS with proper configuration
CORS(app, resources={
    r"/*": {
        "origins": [
            "https://init-12295.web.app",
            "https://cossotconnect.com",
            "http://localhost:5000",
            "http://localhost:3000"
        ],
        "methods": ["GET", "POST", "PUT", "DELETE", "OPTIONS"],
        "allow_headers": ["Content-Type", "Authorization"],
        "expose_headers": ["Content-Range", "X-Content-Range"],
        "supports_credentials": True,
        "max_age": 3600
    }
})

# Set absolute paths for template and static folders
app.template_folder = os.path.abspath('templates')
app.static_folder = os.path.abspath('static')
app.static_url_path = '/static'

if __name__ == '__main__':
    port = int(os.environ.get('PORT', 10000))
    app.run(host='0.0.0.0', port=port) 