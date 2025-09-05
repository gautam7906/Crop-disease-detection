# Gunicorn configuration file for memory optimization

# Server socket
bind = "0.0.0.0:8000"
backlog = 2048

# Worker processes
workers = 1  # Use only 1 worker to save memory
worker_class = "sync"
worker_connections = 1000
timeout = 120
keepalive = 2

# Memory management
max_requests = 100  # Restart worker after 100 requests to prevent memory leaks
max_requests_jitter = 10
preload_app = False  # Don't preload app to save startup memory

# Logging
loglevel = "info"
accesslog = "-"
errorlog = "-"

# Process naming
proc_name = 'crop_disease_app'

# Server mechanics
daemon = False
pidfile = None
user = None
group = None
tmp_upload_dir = None

# Memory limits
limit_request_line = 4094
limit_request_fields = 100
limit_request_field_size = 8190

# Performance tuning for low memory
worker_tmp_dir = "/dev/shm"  # Use shared memory for temporary files