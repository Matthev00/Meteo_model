from meteo_model.data.api.fetch_data import get_normalised_data_from_api
from pathlib import Path
import os
import torch


def get_weather_tensor_for_days(days: int, location_names: list[str]) -> torch.Tensor:
    normalised_data_from_api = get_normalised_data_from_api(days, location_names)
    loc_days_attr = [df.values.tolist() for df in normalised_data_from_api.values()]
    api_data_tensor = torch.tensor(loc_days_attr, dtype=torch.float32)
    return api_data_tensor



if __name__ == "__main__":
    print(get_weather_tensor_for_days(7, ["WARSAW", "WROCLAW"]))
