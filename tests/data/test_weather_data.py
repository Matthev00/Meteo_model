from hypothesis import given, strategies as st
from meteostat import Point
from meteo_model.station import WeatherStation
from meteo_model.data.weather_data import stations_to_dict


@given(
    locations=st.dictionaries(
        keys=st.text(min_size=1, max_size=20),
        values=st.tuples(st.floats(-90, 90), st.floats(-180, 180)),
        max_size=10,
    )
)
def test_stations_to_dict_hypothesis(locations):
    stations = stations_to_dict(locations)
    assert isinstance(stations, dict)
    assert len(stations) == len(locations)
    for key, station in stations.items():
        assert isinstance(key, str)
        assert isinstance(station, WeatherStation)
        assert isinstance(station.point, Point)
