import mlflow.pytorch
from meteo_model.model.base_model import BaseWeatherModel


def load_model(model_name: str, model_version: int, map_location: str) -> BaseWeatherModel:
    return mlflow.pytorch.load_model(model_uri=f"models:/{model_name}/{model_version}", map_location=map_location)
