from config import LOCATIONS, START_YEAR, END_YEAR, BASE_PATH, STATIONS_CACHE_DIR
from datetime import datetime
from pathlib import Path
from meteostat import Stations
from meteo_model.station import WeatherStation
from meteo_model.data.weather_data import stations_to_dict, get_weather_data
from meteo_model.utils.file_utils import sanitize_filename, prepare_directory, save_data_to_csv

Stations.cache_dir = str(STATIONS_CACHE_DIR)


def collect_and_save_weather_data(stations: dict[str, WeatherStation], start_year: int, end_year: int, base_path: Path):
    """Collect and save weather data for each year in the given range."""
    for year in range(start_year, end_year + 1):
        year_path = base_path / str(year)
        prepare_directory(year_path)

        for station, station_data in stations.items():
            sanitized_station = sanitize_filename(station)
            weather_data = get_weather_data({station: station_data}, datetime(year, 1, 1), datetime(year, 12, 31))
            if not weather_data.empty:
                file_path = year_path / f"{sanitized_station}_weather_data.csv"
                save_data_to_csv(weather_data, file_path)


def process_location_data(lat: float, lon: float, start_year: int, end_year: int, base_path: Path) -> None:
    """
    Collects and saves weather data for a specific location and time range.
    """
    stations_dict = stations_to_dict(LOCATIONS)
    collect_and_save_weather_data(stations_dict, START_YEAR, END_YEAR, Path(BASE_PATH))


def main():
    stations_dict = stations_to_dict(LOCATIONS)
    collect_and_save_weather_data(stations_dict, START_YEAR, END_YEAR, Path(BASE_PATH))


if __name__ == "__main__":
    main()
