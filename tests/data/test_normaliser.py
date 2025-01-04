import pytest
import pandas as pd
import numpy as np
from hypothesis import given, assume, strategies as st
from meteo_model.data.normaliser import (
    _normalise_prcp,
    _normalise_snow,
    _normalise_wdir,
    _inverse_normalize_norm_like,
    _inverse_prcp,
    _inverse_snow,
    _inverse_wdir,
    normalize_data,
    inverse_normalize_data,
)


@pytest.fixture
def sample_stats():
    return {
        "tavg": {"mean": 10.0, "std": 2.0},
        "tmin": {"mean": 5.0, "std": 1.5},
        "tmax": {"mean": 15.0, "std": 3.0},
        "wspd": {"mean": 5.0, "std": 1.0},
        "pres": {"mean": 1013.0, "std": 5.0},
        "snow": {"mean": 0.5}
    }


@given(data=st.lists(st.floats(min_value=0, max_value=100), min_size=10, max_size=10))
def test_normalise_prcp(data):
    series = pd.Series(data)
    normalised = _normalise_prcp(series)
    assert all(normalised >= 0)


@given(
    data=st.lists(st.floats(min_value=0, max_value=100), min_size=10, max_size=10),
    stats=st.fixed_dictionaries({"mean": st.floats(min_value=0.1, max_value=10)})
)
def test_normalise_snow(data, stats):
    series = pd.Series(data)
    normalised = _normalise_snow(series, {"snow": stats})
    assert all(normalised >= 0)

@given(data=st.lists(st.floats(min_value=0, max_value=360), min_size=10, max_size=10))
def test_normalise_wdir(data):
    series = pd.Series(data)
    sin, cos = _normalise_wdir(series)
    assert all(-1 <= sin) and all(sin <= 1)
    assert all(-1 <= cos) and all(cos <= 1)


@given(
    data=st.lists(st.floats(min_value=-100, max_value=100), min_size=10, max_size=10),
    stats=st.fixed_dictionaries({"mean": st.floats(min_value=-50, max_value=50), "std": st.floats(min_value=0.1, max_value=10)})
)
def test_inverse_normalize_norm_like(data, stats):
    series = pd.Series(data)
    inv = _inverse_normalize_norm_like(series, "tavg", {"tavg": stats})
    assert inv.equals((series * stats["std"]) + stats["mean"])


@given(data=st.lists(st.floats(min_value=0, max_value=100), min_size=10, max_size=10))
def test_inverse_prcp(data):
    series = pd.Series(data)
    normalised = _normalise_prcp(series)
    inv = _inverse_prcp(normalised)
    assert np.allclose(inv, series)


@given(
    data=st.lists(st.floats(min_value=0, max_value=100), min_size=10, max_size=10),
    stats=st.fixed_dictionaries({"mean": st.floats(min_value=0.1, max_value=10)})
)
def test_inverse_snow(data, stats):
    series = pd.Series(data)
    normalised = _normalise_snow(series, {"snow": stats})
    inv = _inverse_snow(normalised, {"snow": stats})
    assert np.allclose(inv, series)


@given(
    angles=st.lists(st.floats(min_value=0, max_value=360), min_size=10, max_size=10)
)
def test_inverse_wdir(angles):
    series = pd.Series(angles)
    sin, cos = _normalise_wdir(series)
    inv_wdir = _inverse_wdir(sin, cos)

    assume(np.allclose(np.sqrt(sin**2 + cos**2), 1.0, atol=1e-6))

    assert np.allclose(inv_wdir, series, atol=1e-6)


def test_normalize_data(sample_stats):
    df = pd.DataFrame({
        "tavg": [10.0, 12.0],
        "tmin": [5.0, 6.0],
        "tmax": [15.0, 14.0],
        "wspd": [5.0, 4.0],
        "pres": [1013.0, 1018.0],
        "prcp": [0.0, 1.0],
        "snow": [0.0, 1.0],
        "wdir": [90, 180]
    })
    norm_df = normalize_data(df, sample_stats)
    assert norm_df.shape == (2, len(df.columns) + 1)


def test_inverse_normalize_data(sample_stats):
    df = pd.DataFrame({
        "tavg": [0.0, 1.0],
        "tmin": [0.0, 1.0],
        "tmax": [0.0, 1.0],
        "wspd": [0.0, 1.0],
        "pres": [0.0, 1.0],
        "prcp": [0.0, 1.0],
        "snow": [0.0, 1.0],
        "sin_wdir": [0.0, 1.0],
        "cos_wdir": [1.0, 0.0]
    })
    inv_df = inverse_normalize_data(df, sample_stats)

    expected_columns = ["tavg", "tmin", "tmax", "wspd", "pres", "prcp", "snow", "wdir"]
    assert sorted(inv_df.columns) == sorted(expected_columns)

    assert inv_df.shape == (2, len(expected_columns))
