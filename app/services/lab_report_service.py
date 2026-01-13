import os
import shutil
from config.lab_report_config import CLOUDINARY_API_KEY, CLOUDINARY_API_SECRET,CLOUDINARY_CLOUD_NAME,OPENAI_API_KEY,SYSTEM_PROMPT,LOG_FILE,LOG_FORMAT,DATE_FORMAT
from cloudinary.uploader import upload
from langchain_openai import ChatOpenAI
from app.schemas.schema import ReportState, WellnessReport, MedicalReportClassify
from langchain_openai import ChatOpenAI
from langchain_core.messages import HumanMessage, SystemMessage
from langgraph.graph import StateGraph, START, END
import os
import shutil
import json
from cloudinary.uploader import upload
import base64
import logging

logging.basicConfig(
    filename=LOG_FILE,
    format=LOG_FORMAT,
    datefmt=DATE_FORMAT,
    level=logging.INFO
)
logger = logging.getLogger(__name__)

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