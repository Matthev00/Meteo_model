import os
import re
from meteostat import Point, Stations, Daily
from datetime import datetime
import pandas as pd


def get_nearest_stations(lat: float, lon: float, limit: int = 5) -> pd.DataFrame:
    """
    Get the nearest weather stations to a location
    """
    return Stations().nearby(lat, lon).fetch(limit=limit)


def stations_to_dict(stations: pd.DataFrame) -> dict[str, Point]:
    """
    Convert a DataFrame of stations to a dictionary with station names as keys
    and Points (latutude, lontitude, altitude) as values
    """
    return {
        row['name']: Point(row['latitude'], row['longitude'], row['elevation'])
        for _, row in stations.iterrows()
    }


def get_weather_data(stations: dict[str, Point], start: datetime, end: datetime) -> pd.DataFrame:
    """
    Get weather data for a list of stations
    """
    data = pd.concat([
        Daily(point, start, end).fetch().assign(station=station)
        for station, point in stations.items()
    ], ignore_index=True)
    return data


def save_weather_data(data: pd.DataFrame, path: str):
    """
    Save weather data to a CSV file
    """
    data.to_csv(path, index=False)


def sanitize_filename(name: str) -> str:
    """
    Sanitize the station name to make it safe for file names.
    """
    sanitized_name = re.sub(r'[^\w\s-]', '', name).replace(" ", "_").replace("/", "_")
    return re.sub(r'_+', '_', sanitized_name).strip("_")


def prepare_directory(path: str):
    """
    Ensure the directory exists, creating it if necessary.
    """
    os.makedirs(path, exist_ok=True)


def collect_and_save_weather_data(stations: dict[str, Point], start_year: int, end_year: int, base_path: str):
    """
    Collect and save weather data for each year in the given range
    """
    for year in range(start_year, end_year + 1):
        year_path = os.path.join(base_path, str(year))
        prepare_directory(year_path)

        for station, point in stations.items():
            sanitized_station = sanitize_filename(station)
            weather_data = get_weather_data({station: point}, datetime(year, 1, 1), datetime(year, 12, 31))
            file_path = os.path.join(year_path, f"{sanitized_station}_weather_data{year}.csv")
            save_weather_data(weather_data, file_path)


def main():
    WARSAW_LAT, WARSAW_LON = 52.23, 21.01
    START_YEAR, END_YEAR = 2000, 2020
    BASE_PATH, STATIONS_CACHE_DIR = "data/raw/weather_data", "data/cache"

    Stations.cache_dir = STATIONS_CACHE_DIR
    stations_df = get_nearest_stations(WARSAW_LAT, WARSAW_LON)
    stations_dict = stations_to_dict(stations_df)
    collect_and_save_weather_data(stations_dict, START_YEAR, END_YEAR, BASE_PATH)


if __name__ == "__main__":
    main()
