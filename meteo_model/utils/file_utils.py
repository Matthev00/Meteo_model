import re
import pandas as pd
from pathlib import Path


def sanitize_filename(name: str) -> str:
    """Sanitize the station name to make it safe for file names."""
    sanitized_name = re.sub(r'[^\w\s-]', '', name).replace(" ", "_").replace("/", "_")
    return re.sub(r'_+', '_', sanitized_name).strip("_")


def prepare_directory(path: Path):
    """Ensure the directory exists, creating it if necessary."""
    path.mkdir(parents=True, exist_ok=True)


def save_data_to_csv(data: pd.DataFrame, path: Path):
    """Save weather data to a CSV file."""
    data.to_csv(path, index=False)


def get_station_name_from_city_name(city_name: str) -> str:
    """Get the station name from the city name."""
    staions = {
        "Warszawa": "WARSAW",
        "Krakow": "KRAKOW",
        "Wroclaw": "WROCLAW",
        "Poznan": "POZNAN",
        "Bialystok": "BIALYSTOK",
    }
    return staions.get(city_name, city_name)