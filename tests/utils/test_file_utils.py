import pytest
from hypothesis import given, settings, strategies as st
from tempfile import TemporaryDirectory
from pathlib import Path
import logging
import pandas as pd
from meteo_model.utils.file_utils import sanitize_filename, prepare_directory, save_data_to_csv, get_station_name_from_city_name


@given(name=st.text())
@settings(max_examples=100)
def test_sanitize_filename_random(name):
    sanitized = sanitize_filename(name)
    assert "/" not in sanitized
    assert " " not in sanitized
    assert not sanitized.startswith("_")
    assert not sanitized.endswith("_")


@pytest.mark.parametrize(
    "input_name,expected",
    [
        ("Station/Name", "Station_Name"),
        ("Station  Name", "Station_Name"),
        ("__Station__", "Station"),
        ("Sta@tion!", "Station"),
        ("", ""),
    ]
)
def test_sanitize_filename_cases(input_name, expected):
    assert sanitize_filename(input_name) == expected


@given(
    subdir=st.text(
        min_size=1,
        max_size=100,
        alphabet="abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789_-/"
    ).filter(lambda s: not s.startswith("/") and ".." not in s)
)
def test_prepare_directory_creates_new_directory(subdir):
    with TemporaryDirectory() as tempdir:
        path = Path(tempdir) / subdir
        prepare_directory(path)
        assert path.exists() and path.is_dir()


@given(depth=st.integers(min_value=1, max_value=10), name=st.text(
    min_size=1,
    max_size=10,
    alphabet="abcdefghijklmnopqrstuvwxyz"
))
def test_prepare_directory_creates_nested_directories(depth, name):
    with TemporaryDirectory() as tempdir:
        subdirs = "/".join([name] * depth)
        path = Path(tempdir) / subdirs
        prepare_directory(path)
        assert path.exists() and path.is_dir()


def test_prepare_directory_already_exists():
    with TemporaryDirectory() as tempdir:
        path = Path(tempdir)
        assert path.exists() and path.is_dir()
        prepare_directory(path)
        assert path.exists() and path.is_dir()


@given(
    subdir=st.text(
        min_size=1,
        max_size=100,
        alphabet="abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789_-/"
    ).filter(lambda s: not s.startswith("/") and ".." not in s)
)
def test_prepare_directory_multiple_calls(subdir):
    with TemporaryDirectory() as tempdir:
        path = Path(tempdir) / subdir
        prepare_directory(path)
        assert path.exists() and path.is_dir()
        prepare_directory(path)
        assert path.exists() and path.is_dir()


@given(
    subdir=st.text(
        min_size=1,
        max_size=50,
        alphabet="abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789_-/"
    ).filter(lambda s: not s.startswith("/") and ".." not in s)
)
def test_prepare_directory_relative_and_absolute_paths(subdir):
    with TemporaryDirectory() as tempdir:
        relative_path = Path(tempdir) / subdir
        prepare_directory(relative_path)
        assert relative_path.exists() and relative_path.is_dir()

        absolute_path = relative_path.resolve()
        prepare_directory(absolute_path)
        assert absolute_path.exists() and absolute_path.is_dir()


def test_prepare_directory_permission_error():
    with pytest.raises(PermissionError):
        path = Path("/root/forbidden_directory")
        prepare_directory(path)


@given(subdir=st.text(alphabet="!@#$%^&*()<>|", min_size=1, max_size=100))
def test_prepare_directory_invalid_path_hypothesis(subdir):
    with TemporaryDirectory() as tempdir:
        path = Path(tempdir) / subdir
        with pytest.raises(OSError):
            prepare_directory(path)


def test_prepare_directory_logs_permission_error(caplog):
    with pytest.raises(PermissionError):
        path = Path("/root/forbidden_directory")
        prepare_directory(path)

    assert "PermissionError" in caplog.text
    assert "/root/forbidden_directory" in caplog.text


def test_prepare_directory_logs_os_error(caplog):
    with caplog.at_level(logging.ERROR):
        with pytest.raises(OSError):
            path = Path("/invalid_path_with_illegal_characters?<>|")
            prepare_directory(path)

    assert "OSError" in caplog.text
    assert "Path contains invalid characters" in caplog.text
    assert "/invalid_path_with_illegal_characters?<>|" in caplog.text


def test_save_data_to_csv_valid_data():
    data = pd.DataFrame({
        "Temperature": [15.0, 20.5, 25.1],
        "Pressure": [1015, 1020, 1025]
    })

    with TemporaryDirectory() as tempdir:
        path = Path(tempdir) / "weather_data.csv"
        save_data_to_csv(data, path)

        assert path.exists()

        loaded_data = pd.read_csv(path)
        pd.testing.assert_frame_equal(data, loaded_data)


def test_save_data_to_csv_empty_data():
    data = pd.DataFrame()

    with pytest.raises(ValueError, match="DataFrame is empty. Cannot save to CSV."):
        with TemporaryDirectory() as tempdir:
            path = Path(tempdir) / "weather_data.csv"
            save_data_to_csv(data, path)


@given(city_name=st.sampled_from(["Warszawa", "Krakow", "Wroclaw", "Poznan", "Bialystok"]))
def test_get_station_name_from_city_name_known(city_name):
    result = get_station_name_from_city_name(city_name)
    assert result.isupper()


@given(city_name=st.text(min_size=1, max_size=100).filter(lambda x: x.lower() not in ["warszawa", "krakow", "wroclaw", "poznan", "bialystok"]))
def test_get_station_name_from_city_name_unknown(city_name):
    result = get_station_name_from_city_name(city_name)
    assert result == city_name
