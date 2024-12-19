from meteo_model.data.api.fetch_data import get_normalised_data_from_api
from meteo_model.data.config import LOCATIONS_NAMES
import torch


def get_weather_tensor_for_days(days: int, location_names: list[str]) -> tuple[torch.Tensor, str]:
    normalised_data_from_api, end_day = get_normalised_data_from_api(days, location_names)
    loc_days_attr = [df.values.tolist() for df in normalised_data_from_api.values()]
    api_data_tensor = torch.tensor(loc_days_attr, dtype=torch.float32)
    return api_data_tensor, end_day


if __name__ == "__main__":
    print(get_weather_tensor_for_days(7, LOCATIONS_NAMES))
