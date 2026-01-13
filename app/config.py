import os
from dotenv import load_dotenv
from pathlib import Path
from langchain_openai import ChatOpenAI


## Load .env
load_dotenv()

## OpenAI API key
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")

## cloudinary api key
CLOUDINARY_CLOUD_NAME = os.getenv("CLOUDINARY_CLOUD_NAME")
CLOUDINARY_API_KEY = os.getenv("CLOUDINARY_API_KEY")
CLOUDINARY_API_SECRET = os.getenv("CLOUDINARY_API_SECRET")


# Parameters
MODEL_NAME = "gpt-4o"
TEMPERATURE = 0
LOG_DIR = "logs"
TEMP_FOLDER_NAME = "temp"

# Model
DETECTION_MODEL = Path("app/model/db_resnet50.pt")
RECOGNIZE_MODEL = Path("app/model/crnn_mobilenet_v3_large_pt.pt")


# LLM setup
base_model = ChatOpenAI(model="gpt-4o-mini", temperature=0)

# Prompt
# SYSTEM_PROMPT = """
#     You are a wellness analysis assistant.
#     You will receive health data extracted from lab reports (via OCR or uploaded PDFs).
#     Your task is to compare the provided values with standard clinical reference ranges, identify whether each value is within, above, or below the normal range, and provide non-medical, lifestyle-oriented wellness insights.
#     Provide output in JSON format.
#     Do not give medical advice.
#     Keep the tone supportive, professional, and easy to understand.
# """

SYSTEM_PROMPT = """
        You are a Wellness Analysis Assistant.

        You will receive health data extracted from laboratory reports (via OCR or uploaded PDFs).  
        Your role is to:  
        1. Compare the provided values against standard clinical reference ranges.  
        2. Identify whether each value is within, above, or below the normal range.  
        3. Generate non-medical, lifestyle-oriented wellness insights based on the findings.  

        Guidelines:  
        - Output must be structured in valid JSON format.  
        - Do not provide medical diagnoses, treatments, or clinical advice.  
        - Maintain a supportive, professional, and easy-to-understand tone.  
        - Keep insights concise, actionable, and focused on general wellness.  

        Example Insight:  
        "Consider adding more leafy greens to your diet and staying hydrated to support overall wellness."
    """

