from fastapi import FastAPI
import uvicorn
from app.api.v1.endpoints import lab_report
from app.api.v1.endpoints import meal_scaner
from fastapi.middleware.cors import CORSMiddleware

app = FastAPI()

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Router Include
app.include_router(lab_report.router)
app.include_router(meal_scaner.router)


@app.get("/")
async def root():
    return {"messages" : "Suraj AI API documentation"}


if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)
