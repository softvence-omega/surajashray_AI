from langgraph.graph import StateGraph, START, END
from .meal_analyzer import analyze_food
from ..schemas.mealstate import MealState

graph = StateGraph(MealState)
graph.add_node("analyze_food", analyze_food)
graph.add_edge(START, "analyze_food")
graph.add_edge("analyze_food", END)

compiled_graph = graph.compile()