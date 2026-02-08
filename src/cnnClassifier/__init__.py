"""
Step 1 : Create logging code 
"""


import os
import sys
import logging

# Log message format with timestamp, severity level, module name, and message
logging_str = "[%(asctime)s: %(levelname)s: %(module)s: %(message)s:]"

# Directory where log files will be stored
log_dir = "logs"

# Create logs directory if it does not already exist
os.makedirs(log_dir, exist_ok=True)

# Full path to the log file
log_filepath = os.path.join(log_dir, "running_logs.log")

# Configure root logger settings
logging.basicConfig(
    level=logging.INFO,                 # Log INFO level and above
    format=logging_str,                 # Apply custom log format
    handlers=[
        logging.FileHandler(log_filepath, mode="a"),  # Save logs to file (append mode)
        logging.StreamHandler(sys.stdout)              # Also display logs in console
    ]
)

# Create a named logger for the project
logger = logging.getLogger("cnnClassifierLogger")
