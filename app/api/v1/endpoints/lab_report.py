from fastapi import APIRouter, HTTPException, UploadFile, File, BackgroundTasks
from fastapi.responses import JSONResponse
from langchain_openai import ChatOpenAI
import shutil
import json
import os
import datetime
from config.lab_report_config import MODEL_NAME,TEMPERATURE,OPENAI_API_KEY,TEMP_FOLDER_NAME
from app.services.lab_report_service import LabReportGraph, pdf_to_text_openai, cloudinary_file_upload, img_to_text_openai,delete_file,logger
router = APIRouter()

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