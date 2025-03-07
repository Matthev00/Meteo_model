import os
from datetime import datetime, timedelta
import requests
import pandas as pd
from time import sleep
from meteo_model.data.data_cleaner import DataCleanerFromDict
from meteo_model.data.config import PATH_TO_STATS
from meteo_model.data.normaliser import normalize_data
import json
from meteo_model.utils.api_utils import load_env
from meteo_model.data.config import LOCATIONS, API_URL


def get_request_headers() -> dict[str, str]:
    load_env()
    API_KEY = os.getenv("RAPIDAPI_KEY")
    if not API_KEY:
        raise ValueError("API key not found in environment variables.")
    return {"x-rapidapi-key": API_KEY, "x-rapidapi-host": "meteostat.p.rapidapi.com"}


def fetch_weather_data(location: str, start_date: str, end_date: str) -> list:
    if location not in LOCATIONS:
        print(f"Location does not exist in LOCATIONS config.")
        return []

    querystring = {
        "lat": str(LOCATIONS[location][0]),
        "lon": str(LOCATIONS[location][1]),
        "start": start_date,
        "end": end_date,
    }

    response = requests.get(API_URL, headers=get_request_headers(), params=querystring)

    if response.status_code == 200:
        return response.json().get("data", [])
    else:
        print(f"Failed to fetch data for {location}. Status code: {response.status_code}")
        return []


def get_datetimes(days: int) -> tuple[str, str]:
    today = datetime.now()
    end_date = today.strftime("%Y-%m-%d")
    start_date = (today - timedelta(days=days)).strftime("%Y-%m-%d")
    return start_date, end_date


def get_weather_data_for_days(days: int, location_names: list[str]) -> dict[str, list]:
    start_date, end_date = get_datetimes(days)
    data = {}
    for location in location_names:
        data[location] = fetch_weather_data(location, start_date, end_date)
        sleep(0.5)
    return data


def transform_dict_into_df(weather_data) -> pd.DataFrame:
    return pd.DataFrame(weather_data)


def get_start_day_number(date_str: str) -> int:
    date = datetime.strptime(date_str, "%Y-%m-%d %H:%M:%S")
    start_of_year = datetime(date.year, 1, 1)
    day_number = (date - start_of_year).days
    return day_number


def clean_api_data(api_data: dict[str, pd.DataFrame]) -> dict[str, pd.DataFrame]:
    for city, df in api_data.items():
        day_number = get_start_day_number(df["date"].iloc[0])
        cleaner = DataCleanerFromDict(df, city, day_number)
        cleaner.get_cleaned_df()
    return api_data


def read_stat_json() -> dict:
    with open(PATH_TO_STATS) as f:
        stats = json.load(f)
    return stats


def normalise_cleaned_api_data(
    cleaned_api_data: dict[str, pd.DataFrame]
) -> dict[str, pd.DataFrame]:
    stats = read_stat_json()
    normalised_dict = dict()
    for city, cleaned_df in cleaned_api_data.items():
        normalised_df = normalize_data(cleaned_df, stats)
        normalised_dict[city] = normalised_df
    return normalised_dict


def get_date(df_dict: dict[str, pd.DataFrame]) -> str:
    key = next(iter(df_dict))
    return df_dict[key]["date"].iloc[-1]


def get_normalised_data_from_api(
    days: int, location_names: list[str]
) -> tuple[dict[str, pd.DataFrame], str]:
    weather_data_dict = get_weather_data_for_days(days, location_names)
    weather_data_df_dict = {
        city: transform_dict_into_df(weather_data_of_city)
        for city, weather_data_of_city in weather_data_dict.items()
    }
    cleaned = clean_api_data(weather_data_df_dict)
    normalised = normalise_cleaned_api_data(cleaned)
    return normalised, get_date(cleaned)


if __name__ == "__main__":
    cities = ["WARSAW", "BIALYSTOK"]
    weather_data, start_day = get_normalised_data_from_api(7, cities)
    print(start_day)
