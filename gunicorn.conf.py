import os

port = os.getenv('PORT', '10000')
bind = f"0.0.0.0:{port}"
workers = 1