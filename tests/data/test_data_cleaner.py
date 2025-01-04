import pytest
import pandas as pd
import numpy as np
from pathlib import Path
from hypothesis import given, strategies as st
from great_expectations.dataset import PandasDataset
from meteo_model.data.data_cleaner import DataCleaner, DataCleanerAndSaver


@pytest.fixture
def sample_data():
    df1 = pd.DataFrame({
        "station": ["A", "B"],
        "tavg": [10.0, np.nan],
        "tmin": [5.0, np.nan],
        "tmax": [15.0, np.nan],
        "prcp": [np.nan, 2.0],
        "snow": [0.0, np.nan],
        "pres": [1015, np.nan],
        "wdir": [90, np.nan],
        "wspd": [5.0, np.nan],
        "tsun": [np.nan, np.nan],
        "wpgt": [np.nan, np.nan],
    })
    df2 = pd.DataFrame({
        "station": ["C", "D"],
        "tavg": [np.nan, np.nan],
        "tmin": [np.nan, 3.0],
        "tmax": [np.nan, 8.0],
        "prcp": [np.nan, np.nan],
        "snow": [np.nan, np.nan],
        "pres": [np.nan, np.nan],
        "wdir": [np.nan, np.nan],
        "wspd": [np.nan, np.nan],
        "tsun": [np.nan, np.nan],
        "wpgt": [np.nan, np.nan],
    })
    return [df1, df2]


@pytest.fixture
def mock_median_file(tmp_path):
    median_file = tmp_path / "mock_median.csv"
    pd.DataFrame({
        "prcp": [1.0] * 366,
        "wdir": [180] * 366,
        "wspd": [2.5] * 366,
        "pres": [1013] * 366,
    }).to_csv(median_file, index=False)
    return median_file


def test_drop_columns(sample_data):
    cleaner = DataCleaner(sample_data)
    cleaner.drop_columns()
    for df in cleaner.dataframes:
        assert "station" not in df.columns
        assert "tsun" not in df.columns
        assert "wpgt" not in df.columns


def test_handle_nan_based_on_trend(sample_data):
    cleaner = DataCleaner(sample_data)
    cleaner.handle_NaN_based_on_trend()
    for df in cleaner.dataframes:
        assert df["tavg"].isnull().sum() <= 2


def calculate_median_by_day(self) -> pd.DataFrame:
    """
    Calculate the median values for each day of the year.
    """
    median_by_day = pd.DataFrame(index=range(366), columns=["prcp", "wdir", "wspd", "pres"])

    for day in range(366):
        day_data = [df.iloc[day] for df in self.dataframes if len(df) > day]
        day_df = pd.DataFrame(day_data).dropna(how="all")

        if not day_df.empty:
            median_by_day.loc[day, "prcp"] = (
                day_df["prcp"].median() if "prcp" in day_df else np.nan
            )
            median_by_day.loc[day, "wdir"] = (
                day_df["wdir"].median() if "wdir" in day_df else np.nan
            )
            median_by_day.loc[day, "wspd"] = (
                day_df["wspd"].median() if "wspd" in day_df else np.nan
            )
            median_by_day.loc[day, "pres"] = (
                day_df["pres"].median() if "pres" in day_df else np.nan
            )

    return median_by_day


def test_handle_nan_based_on_seasonal_pattern(sample_data, mock_median_file):
    cleaner = DataCleaner(sample_data)
    cleaner.handle_NaN_based_on_sesonal_pattern(mock_median_file)
    for df in cleaner.dataframes:
        assert df["prcp"].isnull().sum() == 0


def test_clip_snow(sample_data):
    for df in sample_data:
        df["snow"] = df["snow"].fillna(0)
    cleaner = DataCleaner(sample_data)
    cleaner.clip_snow()
    for df in cleaner.dataframes:
        assert (df["snow"] <= 800).all()


def test_save_data(sample_data, tmp_path):
    paths = [tmp_path / f"raw_{i}.csv" for i in range(len(sample_data))]
    cleaner = DataCleanerAndSaver(sample_data, paths)
    cleaner.save_data()
    for path in paths:
        assert Path(str(path).replace("raw", "processed")).exists()


@given(
    tavg=st.lists(st.floats(allow_nan=True, allow_infinity=False), min_size=10, max_size=10)
)
def test_interpolation_hypothesis(tavg):
    df = pd.DataFrame({
        "tavg": tavg,
        "tmin": tavg,
        "tmax": tavg,
        "prcp": tavg,
        "snow": tavg,
        "pres": tavg,
        "wdir": tavg,
        "wspd": tavg,
    })
    cleaner = DataCleaner([df])
    cleaner.handle_NaN_based_on_trend()
    assert df["tavg"].isnull().sum() <= 2


def test_great_expectations(sample_data):
    df = sample_data[0]
    dataset = PandasDataset(df)
    dataset.expect_column_to_exist("tavg")
    dataset.expect_column_values_to_not_be_null("tavg")
    dataset.expect_column_median_to_be_between("tavg", min_value=5, max_value=15)
