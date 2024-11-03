from dataclasses import dataclass
from meteostat import Point


@dataclass
class WeatherStation:
    name: str
    point: Point
