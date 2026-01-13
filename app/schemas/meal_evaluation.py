from pydantic import BaseModel, Field, model_validator
from fastapi import HTTPException

class Nutrition(BaseModel):
    calories: int = Field(description="Measure the calories")
    protein_g: int = Field(description="Measure the protein in grams")
    carbs_g: int = Field(description="Measure the carbs in grams")
    fats_g: int = Field(description="Measure the fats in grams")
    is_meal: bool = Field(description="True if this is a meal image, False otherwise")

class Evaluation(BaseModel):
    meal_name: str = Field(description="Name or description of the meal/food item")
    nutrition: Nutrition = Field(description="Nutritional information of the meal")
