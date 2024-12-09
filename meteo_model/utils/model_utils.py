import mlflow.pytorch
import torch
from meteo_model.model.base_model import BaseWeatherModel


def load_model(model_name: str, model_version: int) -> BaseWeatherModel:
    return mlflow.pytorch.load_model(model_uri=f"models:/{model_name}/{model_version}")
