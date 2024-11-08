from meteostat import Daily, Point
from datetime import datetime
from meteo_model.station import WeatherStation
import pandas as pd


def stations_to_dict(locations: dict[str, tuple[float, float]]) -> dict[str, WeatherStation]:
    """
    Convert a dictionary of locations to a dictionary of WeatherStation objects.
    """
    return {
        name: WeatherStation(name, Point(lat, lon))
        for name, (lat, lon) in locations.items()
    }


def get_weather_data(stations: dict[str, WeatherStation], start: datetime, end: datetime) -> pd.DataFrame:
    """Get weather data for a list of stations."""
    data_frames = [
        Daily(station.point, start, end).fetch().assign(station=station_name)
        for station_name, station in stations.items()
    ]
    return pd.concat(data_frames, ignore_index=True) if data_frames else pd.DataFrame()
