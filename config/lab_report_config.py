import os
import cloudinary
from cloudinary.uploader import upload
import logging
from datetime import datetime


from dotenv import load_dotenv
load_dotenv()
# ==================== CONFIGURATION ====================
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
if not OPENAI_API_KEY:
    raise ValueError("❌ OPENAI_API_KEY not found! Please set it in .env file or environment variable")

MODEL_NAME = "gpt-4o"  # or "gpt-4-turbo" or "gpt-3.5-turbo"
TEMPERATURE = 0.3
TEMP_FOLDER_NAME = "temp_files"
LOG_DIR = "logs"

CLOUDINARY_CLOUD_NAME = os.getenv("CLOUDINARY_CLOUD_NAME")
CLOUDINARY_API_KEY = os.getenv("CLOUDINARY_API_KEY")
CLOUDINARY_API_SECRET = os.getenv("CLOUDINARY_API_SECRET")

SYSTEM_PROMPT = """You are a medical report analyzer. Extract information from medical/lab reports (including ECG, X-ray, blood tests, etc.) and provide wellness insights.
Focus on:
- Patient demographics
- Test type (ECG, blood test, imaging, etc.)
- Key findings and measurements
- Normal ranges or expected values
- Whether findings are within normal limits
- Practical, lifestyle-oriented wellness advice (not medical advice)

For ECG reports:
- Heart rate (BPM)
- Rhythm (Normal Sinus Rhythm, etc.)
- Any abnormalities detected
- PR interval, QRS duration, QT interval if available"""

# ==================== CLOUDINARY SETUP ====================
if CLOUDINARY_CLOUD_NAME and CLOUDINARY_API_KEY and CLOUDINARY_API_SECRET:
    cloudinary.config(
        cloud_name=CLOUDINARY_CLOUD_NAME,
        api_key=CLOUDINARY_API_KEY,
        api_secret=CLOUDINARY_API_SECRET
    )
    print("✅ Cloudinary configured successfully")
else:
    print("⚠️ Cloudinary credentials not found. File upload will be skipped.")

# ==================== LOGGING SETUP ====================
os.makedirs(LOG_DIR, exist_ok=True)
LOG_FILE = os.path.join(LOG_DIR, f"log_{datetime.now().strftime('%y-%m-%d')}.log")
LOG_FORMAT = "%(asctime)s | %(levelname)s | %(message)s"
DATE_FORMAT = "%Y-%m-%d %H:%M:%S"
