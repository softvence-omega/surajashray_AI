# meal_analyzer.py
from fastapi import HTTPException
from ..schemas.mealstate import MealState
from ..schemas.meal_evaluation import Evaluation
from ..config import base_model

# ------------------- Meal Analyzer ------------------- #
try:
    from app.config import base_model  # your structured model
except ImportError:
    base_model = None  # fallback placeholder

if base_model:
    structured_model = base_model.with_structured_output(Evaluation)

async def analyze_food(state: MealState) -> MealState:
    try:
        image_url = f"data:image/jpeg;base64,{state['image_base64']}"
        resp = await structured_model.ainvoke(
            [
                {"role": "system", "content": "You are a nutrition expert."},
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": "Analyze this food image, identify the meal name, and return nutritional values in the schema"},
                        {"type": "image_url", "image_url": {"url": image_url}},
                    ],
                },
            ]
        )
        state["evaluation"] = resp
        return state
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
