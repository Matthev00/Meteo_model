from datetime import datetime
from pathlib import Path
from meteostat import Stations
from meteo_model.station import WeatherStation
from meteo_model.data.weather_data import get_nearest_stations, stations_to_dict, get_weather_data
from meteo_model.utils.file_utils import sanitize_filename, prepare_directory, save_data_to_csv


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


def main():
    WARSAW_LAT, WARSAW_LON = 52.23, 21.01
    START_YEAR, END_YEAR = 2000, 2020
    BASE_PATH, STATIONS_CACHE_DIR = Path("data/raw/weather_data"), Path("data/cache")

    Stations.cache_dir = str(STATIONS_CACHE_DIR)
    stations_df = get_nearest_stations(WARSAW_LAT, WARSAW_LON)
    stations_dict = stations_to_dict(stations_df)
    collect_and_save_weather_data(stations_dict, START_YEAR, END_YEAR, BASE_PATH)


if __name__ == "__main__":
    main()
