from fastapi import APIRouter, HTTPException, Depends, UploadFile, File, BackgroundTasks
from fastapi.responses import JSONResponse
import os
import shutil
import json

from app.config import TEMP_FOLDER_NAME
from app.services.lab_report.ocr.ocr import OCR
from app.services.lab_report.graph.graph import LabReportGraph
from app.services.lab_report.llms.onenai_llm import OpenAILLM
from app.utils.helper import delete_file
from app.utils.helper import cloudinary_file_upload


router = APIRouter()

def get_ocr():
    return OCR()

def get_graph():
    llm_model = OpenAILLM()
    llm = llm_model.get_llm_model()
    graph = LabReportGraph(model=llm)
    return graph


@router.post("/lab_report_analysis")
async def lab_report(file : UploadFile = File(...), ocr = Depends(get_ocr), graph = Depends(get_graph), background_task : BackgroundTasks = None):
    
    allowed_file_types = ["image/jpeg","image/png","image/bmp", "application/pdf"]
    
    if file.content_type not in allowed_file_types:
        raise HTTPException(status_code=400, detail="Only Image and PDF file are acceptable.")
    
    # make temp folder 
    os.makedirs(TEMP_FOLDER_NAME, exist_ok=True)
    temp_file_path = os.path.join(TEMP_FOLDER_NAME, file.filename)
    
    try:   
        with open(temp_file_path, "wb") as f:
            shutil.copyfileobj(file.file, f)
            
        # Graph setup
        graph_builder = graph.setup_graph()
            
        ### Classify File
        if file.content_type == "application/pdf":
            print("PDF calling..........")
            
            pdf_data = ocr.pdf_to_text(pdf_file=temp_file_path)
            pdf_text = graph_builder.invoke({
                "report_text" : pdf_data
            })
            
            background_task.add_task(delete_file, TEMP_FOLDER_NAME)
            
            return JSONResponse({
                "file_name" : file.filename,
                "report_text" : json.loads(pdf_text["output"]),
                "file_url" : cloudinary_file_upload(temp_file_path)
            })
        elif file.content_type.startswith("image/"):
            print("Image calling.........")
            
            img_data = ocr.img_to_text(img_file=temp_file_path)
            img_text = graph_builder.invoke({
                "report_text" : img_data
            })
            
            background_task.add_task(delete_file, TEMP_FOLDER_NAME)
            
            return JSONResponse({
                "file_name" : file.filename,
                "report_text" : json.loads(img_text["output"]),
                "file_url" : cloudinary_file_upload(temp_file_path)
            })
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
        