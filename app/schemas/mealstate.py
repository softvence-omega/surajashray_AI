from typing import TypedDict, Optional
from .meal_evaluation import Evaluation

class MealState(TypedDict):
    image_base64: str
    evaluation: Optional[Evaluation]