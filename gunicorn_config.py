import multiprocessing

# Gunicorn configuration
bind = "unix:/tmp/gunicorn.sock"  # Unix socket for Nginx
workers = multiprocessing.cpu_count() * 2 + 1  # Number of worker processes
worker_class = "sync"  # Worker class
timeout = 300  # Timeout in seconds
keepalive = 2  # Keepalive timeout

# Logging
accesslog = "/var/log/gunicorn/access.log"
errorlog = "/var/log/gunicorn/error.log"
loglevel = "info"

# Process naming
proc_name = "ai_clipper"

# SSL (if not terminating SSL at Nginx)
# keyfile = "/path/to/keyfile"
# certfile = "/path/to/certfile"

# Security
limit_request_line = 4094
limit_request_fields = 100
limit_request_field_size = 8190 