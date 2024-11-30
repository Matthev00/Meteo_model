import os
import datetime
import requests
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
    today = datetime.datetime.now()
    end_date = today.strftime("%Y-%m-%d")
    start_date = (today - datetime.timedelta(days=days)).strftime("%Y-%m-%d")
    return start_date, end_date


def get_weather_data_for_days(days: int, location_names: list[str]) -> dict[str, list]:
    start_date, end_date = get_datetimes(days)
    data = {}
    for location in location_names:
        data[location] = fetch_weather_data(location, start_date, end_date)
    return data


if __name__ == "__main__":
    cities = ["WARSAW"]
    weather_data = get_weather_data_for_days(7, cities)
    print(weather_data)
