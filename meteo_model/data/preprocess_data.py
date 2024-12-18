import os
import json
import pandas as pd
from pathlib import Path

from meteo_model.data.get_stats import create_stat_file
from meteo_model.data.normaliser import normalize_data
from meteo_model.data.config import PATH_TO_STATS, LOCATIONS_NAMES, BASE_PATH, MEDIAN_DIR
from meteo_model.utils.file_utils import get_station_name_from_city_name
from meteo_model.data.data_cleaner import DataCleanerAndSaver


def get_raw_data(
    station: str,
    data_dir: Path = Path(BASE_PATH),
    year_range: tuple[int, int] = (2012, 2024),
) -> tuple[list[pd.DataFrame], list[Path]]:
    """
    Load data from csv files and return a list of dataframes.
    """
    data = []
    data_paths = []
    for year in range(year_range[0], year_range[1] + 1):
        file_path = data_dir / str(year) / f"{station}_weather_data.csv"
        if file_path.exists():
            data.append(pd.read_csv(file_path))
            data_paths.append(file_path)
    return data, data_paths


def clean_and_save_data(city_name: str):
    station = get_station_name_from_city_name(city_name)
    data, data_paths = get_raw_data(station)
    yield f"{city_name}: {len(data)} files found."

    cleaner = DataCleanerAndSaver(data, data_paths)
    cleaner.drop_columns()
    yield f"{city_name}: Columns dropped."

    cleaner.handle_NaN_based_on_trend()
    yield f"{city_name}: NaN handled based on trend."
    
    median_dir = Path(MEDIAN_DIR)
    median_dir.mkdir(parents=True, exist_ok=True)
    cleaner.handle_NaN_based_on_sesonal_pattern(median_dir / f"{station}.csv")
    yield f"{city_name}: NaN handled based on seasonal pattern."

    cleaner.clip_snow()
    yield f"{city_name}: Snow values clipped."

    cleaner.save_data()
    yield f"{city_name}: Data saved."


def prepocessing(cities: list[str]):
    clean_data_for_(cities)
    normalize_cleaned_data_for_(cities)


def normalize_cleaned_data_station(
    station,
    stats,
    data_dir: Path = Path("data/processed"),
    year_range: tuple[int, int] = (2012, 2024),
):
    for year in range(year_range[0], year_range[1] + 1):
        input_file_path = data_dir / "weather_data" / str(year) / f"{station}_weather_data.csv"
        norm_year_data_dir = Path(str(data_dir).replace("processed", "normalized")) / str(year)
        if not os.path.exists(norm_year_data_dir):
            os.makedirs(norm_year_data_dir)
        output_file_path = norm_year_data_dir / f"{station}_weather_data.csv"
        if input_file_path.exists():
            cleaned_df = pd.read_csv(input_file_path)
            normalized_df = normalize_data(cleaned_df, stats)
            normalized_df.to_csv(output_file_path, index=False)


def normalize_cleaned_data_for_(
    cities: list[str],
    data_dir: Path = Path("data/processed"),
    year_range: tuple[int, int] = (2012, 2024),
):
    if not Path(PATH_TO_STATS).exists():
        create_stat_file()
    with open(PATH_TO_STATS) as f:
        stats = json.load(f)
    for station in cities:
        normalize_cleaned_data_station(station, stats, data_dir, year_range)


def clean_data_for_(cities: list[str]):
    for city in cities:
        for message in clean_and_save_data(city):
            print(message)
        print(40 * "-")


if __name__ == "__main__":
    cities = LOCATIONS_NAMES
    prepocessing(cities)
