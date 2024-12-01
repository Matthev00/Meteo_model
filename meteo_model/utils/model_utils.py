import mlflow.pytorch
import torch
from meteo_model.model.base_model import BaseWeatherModel
from meteo_model.model.weather_model_tcn import WeatherModelTCN
from meteo_model.utils.evaluation_utils import get_params

def load_model(model_name: str, model_version: int) -> BaseWeatherModel:
    return mlflow.pytorch.load_model(model_uri=f"models:/{model_name}/{model_version}")


def load_tcn_model(path_to_run):
    model = WeatherModelTCN(**get_params(path_to_run))
    model.load_state_dict(torch.load(path_to_run + "artifacts/model_dict/model_dict.pth", weights_only=True))
    model.eval()
    return model
