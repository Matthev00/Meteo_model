from meteostat import Stations, Daily, Point
from datetime import datetime
from meteo_model.station import WeatherStation
import pandas as pd


def get_nearest_stations(lat: float, lon: float, limit: int = 5) -> pd.DataFrame:
    """Get the nearest weather stations to a location."""
    return Stations().nearby(lat, lon).fetch(limit=limit)


def stations_to_dict(stations: pd.DataFrame) -> dict[str, WeatherStation]:
    """Convert a DataFrame of stations to a dictionary with station names as keys."""
    return {
        row['name']: WeatherStation(row['name'], Point(row['latitude'], row['longitude'], row['elevation']))
        for _, row in stations.iterrows()
    }


def get_weather_data(stations: dict[str, WeatherStation], start: datetime, end: datetime) -> pd.DataFrame:
    """Get weather data for a list of stations."""
    data_frames = [
        Daily(station.point, start, end).fetch().assign(station=station_name)
        for station_name, station in stations.items()
    ]
    return pd.concat(data_frames, ignore_index=True) if data_frames else pd.DataFrame()
