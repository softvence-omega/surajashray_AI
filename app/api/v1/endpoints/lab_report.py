from fastapi import APIRouter, HTTPException, UploadFile, File, BackgroundTasks, Query
from fastapi.responses import JSONResponse
from langchain_openai import ChatOpenAI
import shutil
import json
import os
import datetime
from config.lab_report_config import MODEL_NAME,TEMPERATURE,OPENAI_API_KEY,TEMP_FOLDER_NAME
from app.services.lab_report_service import LabReportGraph, pdf_to_text_openai, cloudinary_file_upload, img_to_text_openai,delete_file,logger
import requests
router = APIRouter()


def _try_parse_date(date_str: str):
    """Try to parse a date string into ISO format. Return original if parsing fails."""
    if not date_str:
        return None
    # try ISO first
    try:
        # handle already ISO-like with time
        dt = datetime.datetime.fromisoformat(date_str)
        return dt.isoformat() + "Z"
    except Exception:
        pass

    # common formats to try
    formats = [
        "%Y-%m-%d",
        "%d %b, %Y",
        "%d %b %Y",
        "%d %B, %Y",
        "%d %B %Y",
        "%d-%m-%Y",
        "%m/%d/%Y",
        "%d/%m/%Y",
    ]
    for fmt in formats:
        try:
            dt = datetime.datetime.strptime(date_str, fmt)
            return dt.isoformat() + "Z"
        except Exception:
            continue

    # fallback: return original string
    return date_str

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


@router.post("/lab_report_analysis")
async def lab_report(
    file: UploadFile = File(...),
    access_token: str = Query(..., description="User access token"),
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
        base_url = os.getenv("BASE_URL")
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
            parsed = json.loads(pdf_text["output"])

            file_url = cloudinary_file_upload(temp_file_path)
            background_task.add_task(delete_file, TEMP_FOLDER_NAME)

            # Map to desired response structure
            response_payload = {
                "fileName": file.filename,
                "fileUrl": file_url,
                "reportType": parsed.get("report_type") or parsed.get("test_type") or "OTHER",
                "patientName": parsed.get("patient_name") or None,
                "reportDate": _try_parse_date(parsed.get("report_date")) or None,
                "labName": parsed.get("lab_name") or parsed.get("lab") or None,
                "doctorName": parsed.get("doctor_name") or parsed.get("doctor") or None,
                "reportData": parsed
            }

            # Save to DB
            if base_url:
                db_url = f"{base_url}/api/v1/medical-reports"
                db_headers = {
                    "Authorization": f"Bearer {access_token}",
                    "Content-Type": "application/json"
                }
                try:
                    requests.post(db_url, headers=db_headers, json=response_payload)
                except Exception as e:
                    logger.error(f"Failed to post lab report to DB: {str(e)}")
            return JSONResponse(response_payload)
        elif file.content_type.startswith("image/"):
            print("Image calling.........")
            img_data = img_to_text_openai(img_file=temp_file_path)
            img_text = graph_builder.invoke({
                "report_text": img_data
            })
            parsed = json.loads(img_text["output"])

            file_url = cloudinary_file_upload(temp_file_path)
            background_task.add_task(delete_file, TEMP_FOLDER_NAME)

            response_payload = {
                "fileName": file.filename,
                "fileUrl": file_url,
                "reportType": parsed.get("report_type") or parsed.get("test_type") or "OTHER",
                "patientName": parsed.get("patient_name") or None,
                "reportDate": _try_parse_date(parsed.get("report_date")) or None,
                "labName": parsed.get("lab_name") or parsed.get("lab") or None,
                "doctorName": parsed.get("doctor_name") or parsed.get("doctor") or None,
                "reportData": parsed
            }

            # Save to DB
            if base_url:
                db_url = f"{base_url}/api/v1/medical-reports"
                db_headers = {
                    "Authorization": f"Bearer {access_token}",
                    "Content-Type": "application/json"
                }
                try:
                    requests.post(db_url, headers=db_headers, json=response_payload)
                except Exception as e:
                    logger.error(f"Failed to post lab report to DB: {str(e)}")
            return JSONResponse(response_payload)
    except Exception as e:
        logger.error(f"Error processing file: {e}")
        # Clean up on error
        if os.path.exists(TEMP_FOLDER_NAME):
            shutil.rmtree(TEMP_FOLDER_NAME)
        raise HTTPException(status_code=500, detail=str(e))