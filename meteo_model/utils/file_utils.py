import re
import pandas as pd
from pathlib import Path
import logging


def sanitize_filename(name: str) -> str:
    """Sanitize the station name to make it safe for file names."""
    sanitized_name = re.sub(r"[ /]", "_", name)
    sanitized_name = re.sub(r"[^\w\s-]", "", sanitized_name)
    return re.sub(r"_+", "_", sanitized_name).strip("_")


def prepare_directory(path: Path):
    """Ensure the directory exists, creating it if necessary."""
    invalid_characters = "!@#$%^&*()<>|"
    if any(char in str(path) for char in invalid_characters):
        logging.error(f"OSError: Path contains invalid characters: {path}")
        raise OSError(f"Path contains invalid characters: {path}")
    try:
        path.mkdir(parents=True, exist_ok=True)
    except PermissionError as e:
        logging.error(f"PermissionError: {e}")
        raise
    except OSError as e:
        logging.error(f"OSError: {e}")
        raise


def save_data_to_csv(data: pd.DataFrame, path: Path):
    """Save weather data to a CSV file."""
    if data.empty:
        raise ValueError("DataFrame is empty. Cannot save to CSV.")

    data.to_csv(path, index=False)


def get_station_name_from_city_name(city_name: str) -> str:
    """Get the station name from the city name."""
    staions = {
        "warszawa": "WARSAW",
        "krakow": "KRAKOW",
        "wroclaw": "WROCLAW",
        "poznan": "POZNAN",
        "bialystok": "BIALYSTOK",
    }
    return staions.get(city_name.lower(), city_name)
