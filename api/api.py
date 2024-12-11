from fastapi import FastAPI
from api.predict import router as predict_router

app = FastAPI(title="Meteo Forecasting API")

app.include_router(predict_router, prefix="/predict", tags=["Prediction"])

@app.get("/")
def read_root():
    return {"message": "Welcome to the Meteo Forecasting API"}
