from fastapi import FastAPI, HTTPException, UploadFile, File, BackgroundTasks
from fastapi.responses import JSONResponse
from pydantic import BaseModel, Field
from typing import Literal, Optional, List, TypedDict
from langchain_openai import ChatOpenAI
from langchain_core.messages import HumanMessage, SystemMessage
from langgraph.graph import StateGraph, START, END
import os
import shutil
import json
import logging
from datetime import datetime
import cloudinary
from cloudinary.uploader import upload
import base64
from dotenv import load_dotenv

# Load environment variables from .env file
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

logging.basicConfig(
    filename=LOG_FILE,
    format=LOG_FORMAT,
    datefmt=DATE_FORMAT,
    level=logging.INFO
)
logger = logging.getLogger(__name__)

print(f"✅ OpenAI API Key loaded: {OPENAI_API_KEY[:10]}...")
print(f"✅ Using Model: {MODEL_NAME}")

# ==================== SCHEMAS ====================
class MedicalReportClassify(BaseModel):
    check: Literal["yes", "no"] = Field(..., description="Check if the text/image is a medical report (ECG, blood test, X-ray, CT scan, lab report, etc.) or not. Look for medical terminology, patient info, test results, or medical charts/graphs.")

class LabValue(BaseModel):
    test_name: str = Field(..., description="Name of the lab test or observation")
    value: Optional[float] = Field(None, description="Measured value from lab report, if applicable")
    unit: Optional[str] = Field(None, description="Unit of the measured value, if applicable")
    ref_ranges: Optional[str] = Field(None, description="Standard clinical reference range, if applicable")
    status: Optional[str] = Field(None, description="Whether the value is 'within', 'above', or 'below' the normal range, if applicable")

class WellnessReport(BaseModel):
    patient_name: Optional[str] = Field(None, description="Name of the patient")
    report_date: Optional[str] = Field(None, description="Date of the lab report")
    lab_values: List[LabValue] = Field(..., description="List of lab values or observations with wellness insights")
    wellness_insight: str = Field(..., description="Non-medical, lifestyle-oriented advice based on the value or test")

class ReportState(TypedDict):
    """Represent the structure of the state used in graph"""
    report_text: str
    report_status: str
    output: str

# ==================== UTILITY FUNCTIONS ====================
def delete_file(file_path):
    """Delete temporary files"""
    try:
        if os.path.exists(file_path):
            shutil.rmtree(file_path)
            logger.info(f"Deleted temp folder: {file_path}")
    except Exception as e:
        logger.error(f"Error deleting file: {e}")

def cloudinary_file_upload(file_path):
    """Upload file to Cloudinary"""
    if not all([CLOUDINARY_CLOUD_NAME, CLOUDINARY_API_KEY, CLOUDINARY_API_SECRET]):
        logger.warning("Cloudinary not configured, skipping upload")
        return None
    
    if file_path.lower().endswith(".pdf"):
        try:
            result = upload(
                file=file_path,
                resource_type="auto",
                folder="pdfs"
            )
            return result["secure_url"]
        except Exception as e:
            logger.error(f"Cloudinary upload error: {e}")
            return None
    else:
        try:
            result = upload(
                file=file_path,
                resource_type="auto"
            )
            return result["secure_url"]
        except Exception as e:
            logger.error(f"Cloudinary upload error: {e}")
            return None

# ==================== OCR FUNCTIONS (Using OpenAI Vision) ====================
def pdf_to_text_openai(pdf_file: str) -> str:
    """Extract text from PDF using OpenAI Vision API"""
    try:
        logger.info("Extracting text from PDF using OpenAI Vision...")
        
        # Convert PDF to images and extract text using OpenAI Vision
        from pdf2image import convert_from_path
        
        # Convert PDF pages to images
        images = convert_from_path(pdf_file)
        
        llm = ChatOpenAI(
            model="gpt-4o",
            temperature=0,
            api_key=OPENAI_API_KEY
        )
        
        all_text = []
        for i, image in enumerate(images):
            logger.info(f"Processing PDF page {i+1}/{len(images)}...")
            
            # Convert PIL Image to base64
            from io import BytesIO
            buffered = BytesIO()
            image.save(buffered, format="PNG")
            base64_image = base64.b64encode(buffered.getvalue()).decode('utf-8')
            
            # Extract text using OpenAI Vision
            response = llm.invoke([
                HumanMessage(content=[
                    {
                        "type": "text",
                        "text": "Extract all text from this medical lab report page. Preserve the structure and include all test names, values, units, and reference ranges. Return only the extracted text."
                    },
                    {
                        "type": "image_url",
                        "image_url": {
                            "url": f"data:image/png;base64,{base64_image}"
                        }
                    }
                ])
            ])
            
            all_text.append(response.content)
        
        return " ".join(all_text)
    except Exception as e:
        logger.error(f"Error extracting PDF text: {e}")
        raise ValueError(f"Error extracting PDF text: {e}")

def img_to_text_openai(img_file: str) -> str:
    """Extract text from image using OpenAI Vision API"""
    try:
        logger.info("Extracting text from image using OpenAI Vision...")
        
        # Read image and convert to base64
        with open(img_file, "rb") as image_file:
            base64_image = base64.b64encode(image_file.read()).decode('utf-8')
        
        # Determine image format
        file_extension = img_file.lower().split('.')[-1]
        mime_type = f"image/{file_extension}" if file_extension in ["jpeg", "jpg", "png", "bmp"] else "image/jpeg"
        
        # Use OpenAI Vision to extract text
        llm = ChatOpenAI(
            model="gpt-4o",
            temperature=0,
            api_key=OPENAI_API_KEY
        )
        
        response = llm.invoke([
            HumanMessage(content=[
                {
                    "type": "text",
                    "text": """Analyze this medical report image carefully. This could be:
- ECG/EKG report (with waveform graphs)
- Blood test results
- X-ray or CT scan
- Other medical diagnostic reports

Extract ALL information including:
1. Patient name, age, gender
2. Date of report
3. Type of test/examination
4. All test values, measurements, or findings
5. Reference ranges
6. Any abnormalities or notes
7. For ECG: heart rate, rhythm, intervals, and any abnormal patterns
8. Doctor's observations or impressions

Provide complete details even if some text is small or faint."""
                },
                {
                    "type": "image_url",
                    "image_url": {
                        "url": f"data:{mime_type};base64,{base64_image}"
                    }
                }
            ])
        ])
        
        return response.content
    except Exception as e:
        logger.error(f"Error extracting image text: {e}")
        raise ValueError(f"Error extracting image text: {e}")

# ==================== GRAPH NODES ====================
class ReportNode:
    def __init__(self, model):
        self.llm = model
    
    def classify_report(self, state: ReportState):
        """Determine if the report text is medical and update the state."""
        print("CLASSIFYING REPORT............")
        text = state["report_text"]
        
        try:
            # Enhanced classification prompt
            classification_prompt = f"""Analyze if this is a medical/health report of any kind (ECG, blood test, X-ray, CT scan, MRI, lab results, etc.).

Content to analyze:
{text}

Medical reports can include:
- ECG/EKG with waveform data and heart measurements
- Blood test results with lab values
- Imaging reports (X-ray, CT, MRI, Ultrasound)
- Pathology reports
- Any diagnostic test results

Respond with 'yes' if this appears to be any type of medical/health report, 'no' otherwise."""
            
            llm_with_structure_output = self.llm.with_structured_output(MedicalReportClassify)
            output = llm_with_structure_output.invoke(classification_prompt)
            state["report_status"] = output.check
            
            logger.info(f"Classification result: {output.check}")
            return state
        except Exception as e:
            logger.error(f"Classification error: {e}")
            raise ValueError(e)
    
    def generate_report(self, state: ReportState):
        """Generate a structured JSON wellness report from the report text."""
        print("GENERATING REPORT............")
        input_text = state["report_text"]
        
        try:
            llm_report_structure = self.llm.with_structured_output(WellnessReport)
            output = llm_report_structure.invoke([
                SystemMessage(content=SYSTEM_PROMPT),
                HumanMessage(content=input_text)
            ])
            final_output = output.model_dump_json(indent=4)
            state["output"] = final_output
        except Exception as e:
            fall_back_report = WellnessReport(
                patient_name=None,
                report_date=None,
                lab_values=[],
                wellness_insight="Unable to generate detailed insights. Please ensure the medical report contains valid test results."
            )
            state["output"] = fall_back_report.model_dump_json(indent=4)
            logger.error(f"Validation Error: {e}")
        
        return state
    
    def not_report(self, state: ReportState):
        """Handle cases where the input text is not a valid medical or lab report."""
        print("NOT ACTUAL REPORT............")
        data = "The provided text does not appear to be a medical or lab report. Please upload a valid health report for analysis."
        state["output"] = json.dumps(data, indent=4)
        return state
    
    def router_decision(self, state: ReportState):
        """Decide the next action based on whether the report is medical."""
        if state['report_status'] == "yes":
            return "generate_report"
        else:
            return "not_report"

# ==================== GRAPH SETUP ====================
class LabReportGraph:
    def __init__(self, model):
        self.llm = model
        self.graph_builder = StateGraph(ReportState)
    
    def setup_graph(self):
        try:
            logger.info("Setup Graph Initializing...........")
            report_node = ReportNode(self.llm)
            
            # Add Nodes
            self.graph_builder.add_node("classify_report", report_node.classify_report)
            self.graph_builder.add_node("generate_report", report_node.generate_report)
            self.graph_builder.add_node("not_report", report_node.not_report)
            
            # Add edges
            self.graph_builder.add_edge(START, "classify_report")
            self.graph_builder.add_conditional_edges(
                "classify_report",
                report_node.router_decision,
                {
                    "generate_report": "generate_report",
                    "not_report": "not_report"
                }
            )
            self.graph_builder.add_edge("generate_report", END)
            self.graph_builder.add_edge("not_report", END)
            
            graph = self.graph_builder.compile()
            return graph
        except Exception as e:
            logger.error(f"Graph setup error: {e}")
            raise ValueError(e)

# ==================== API ROUTER ====================
app = FastAPI(title="Lab Report Analyzer API", version="1.0.0")

def get_llm():
    """Initialize OpenAI LLM"""
    logger.info("OpenAI Model Initializing.....")
    llm = ChatOpenAI(
        model=MODEL_NAME,
        temperature=TEMPERATURE,
        api_key=OPENAI_API_KEY
    )
    return llm

def get_graph():
    """Initialize LangGraph"""
    llm = get_llm()
    graph = LabReportGraph(model=llm)
    return graph

@app.get("/")
async def root():
    """Health check endpoint"""
    return {
        "status": "running",
        "message": "Lab Report Analyzer API is working!",
        "openai_configured": bool(OPENAI_API_KEY),
        "cloudinary_configured": bool(CLOUDINARY_CLOUD_NAME)
    }

@app.post("/lab_report_analysis")
async def lab_report(
    file: UploadFile = File(...),
    background_task: BackgroundTasks = BackgroundTasks()
):
    """Main endpoint for lab report analysis"""
    allowed_file_types = ["image/jpeg", "image/png", "image/bmp", "application/pdf"]
    
    if file.content_type not in allowed_file_types:
        raise HTTPException(status_code=400, detail="Only Image and PDF files are acceptable.")
    
    # Make temp folder
    os.makedirs(TEMP_FOLDER_NAME, exist_ok=True)
    temp_file_path = os.path.join(TEMP_FOLDER_NAME, file.filename)
    
    try:
        # Save uploaded file
        with open(temp_file_path, "wb") as f:
            shutil.copyfileobj(file.file, f)
        
        logger.info(f"Processing file: {file.filename}")
        
        # Graph setup
        graph = get_graph()
        graph_builder = graph.setup_graph()
        
        # Classify File and extract text
        if file.content_type == "application/pdf":
            print("PDF calling..........")
            pdf_data = pdf_to_text_openai(pdf_file=temp_file_path)
            pdf_text = graph_builder.invoke({
                "report_text": pdf_data
            })
            
            file_url = cloudinary_file_upload(temp_file_path)
            background_task.add_task(delete_file, TEMP_FOLDER_NAME)
            
            return JSONResponse({
                "file_name": file.filename,
                "report_text": json.loads(pdf_text["output"]),
                "file_url": file_url
            })
        elif file.content_type.startswith("image/"):
            print("Image calling.........")
            img_data = img_to_text_openai(img_file=temp_file_path)
            img_text = graph_builder.invoke({
                "report_text": img_data
            })
            
            file_url = cloudinary_file_upload(temp_file_path)
            background_task.add_task(delete_file, TEMP_FOLDER_NAME)
            
            return JSONResponse({
                "file_name": file.filename,
                "report_text": json.loads(img_text["output"]),
                "file_url": file_url
            })
    except Exception as e:
        logger.error(f"Error processing file: {e}")
        # Clean up on error
        if os.path.exists(TEMP_FOLDER_NAME):
            shutil.rmtree(TEMP_FOLDER_NAME)
        raise HTTPException(status_code=500, detail=str(e))
