from fastapi import APIRouter, File, UploadFile, HTTPException
from fastapi.responses import JSONResponse
import base64

from app.services.meal_graph import compiled_graph
from app.schemas.mealstate import MealState

router = APIRouter()

@router.post("/analyze-food")
async def analyze_food_endpoint(file: UploadFile = File(...)):
    try:
        image_bytes = await file.read()
        if not image_bytes:
            raise HTTPException(status_code=400, detail="Uploaded file is empty.")
        
        image_base64 = base64.b64encode(image_bytes).decode("utf-8")
        result: MealState = await compiled_graph.ainvoke({"image_base64": image_base64, "evaluation": None})
        
        return JSONResponse(content={
            "meal_name": result["evaluation"].meal_name,
            "nutrition": result["evaluation"].nutrition.dict()
        })
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
