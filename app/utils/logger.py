import logging
import os
from datetime import datetime

from app.config import LOG_DIR


# Make DIR
os.makedirs(LOG_DIR, exist_ok=True)

LOG_FILE = os.path.join(LOG_DIR, f"log_{datetime.now().strftime('%y-%m-%d')}.log")
LOG_FORMAT = "%(asctime)s | %(levelname)s | %(message)s"
DATE_FORMAT = "%Y-%m-%d %H:%M:%S"

def get_logger(name : str) -> logging.Logger:
    logger = logging.getLogger(name)
    
    if not logger.handlers:
        
        logging.basicConfig(
            filename=LOG_FILE,
            format=LOG_FORMAT,
            datefmt=DATE_FORMAT,
            level = logging.INFO
        )
    return logger

