import pandas as pd
from pathlib import Path

from meteo_model.utils.file_utils import get_station_name_from_city_name
from meteo_model.data.data_cleaner import DataCleaner


def get_raw_data(
    station: str,
    data_dir: Path = Path("data/raw/weather_data"),
    year_range: tuple[int, int] = (2012, 2024),
) -> list[pd.DataFrame]:
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

    cleaner = DataCleaner(data, data_paths)
    cleaner.drop_columns()
    yield f"{city_name}: Columns dropped."

    cleaner.handle_NaN_based_on_trend()
    yield f"{city_name}: NaN handled based on trend."

    cleaner.handle_NaN_based_on_sesonal_pattern()
    yield f"{city_name}: NaN handled based on seasonal pattern."

    cleaner.clip_snow()
    yield f"{city_name}: Snow values clipped."

    cleaner.save_data()
    yield f"{city_name}: Data saved."


def main():
    cities = ["Bialystok", "Warszawa", "Wroclaw", "Krakow", "Poznan"]
    for city in cities:
        for message in clean_and_save_data(city):
            print(message)
        print(40 * "-")


if __name__ == "__main__":
    main()
