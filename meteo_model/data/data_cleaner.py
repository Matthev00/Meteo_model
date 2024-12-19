import pandas as pd
import numpy as np
from pathlib import Path
import os

from meteo_model.utils.file_utils import prepare_directory
from meteo_model.data.config import MEDIAN_DIR


class DataCleaner:
    def __init__(self, dataframes, columns_to_drop=["station", "tsun", "wpgt"]):
        self.dataframes = dataframes
        self.columns_to_drop = columns_to_drop

    def drop_columns(self) -> None:
        """
        Drop unnecessary columns from the dataframes in place.
        ['station', 'tsun', 'wpgt']
        """
        for df in self.dataframes:
            df.drop(columns=self.columns_to_drop, inplace=True)

    def handle_NaN_based_on_trend(self) -> None:
        """
        Handle missing values in the data based on trend. In Place.
        """
        for df in self.dataframes:
            df["tavg"] = df["tavg"].interpolate(method="linear", limit=2, limit_direction="both")
            df["tmin"] = df["tmin"].interpolate(method="linear", limit=2, limit_direction="both")
            df["tmax"] = df["tmax"].interpolate(method="linear", limit=2, limit_direction="both")
            df["prcp"] = df["prcp"].interpolate(method="nearest", limit=2, limit_direction="both")
            df["snow"] = df["snow"].fillna(0)
            df["pres"] = df["pres"].interpolate(method="linear", limit=2, limit_direction="both")
            df["wdir"] = df["wdir"].interpolate(method="nearest", limit=3, limit_direction="both")
            df["wspd"] = df["wspd"].interpolate(method="linear", limit=2, limit_direction="both")

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

    def save_median_to_file(self, median_by_day: pd.DataFrame, file_path: Path) -> None:
        median_by_day.to_csv(file_path, index=False)

    def load_median_from_file(self, file_path: Path) -> pd.DataFrame:
        return pd.read_csv(file_path)

    def handle_NaN_based_on_sesonal_pattern(
        self, median_file: Path, start_offset: int = 0
    ) -> None:
        """
        Handle missing values in the data based on group. In Place.
        """
        if os.path.exists(median_file):
            median_by_day = self.load_median_from_file(median_file)
        else:
            median_by_day = self.calculate_median_by_day()
            self.save_median_to_file(median_by_day, median_file)
        for df in self.dataframes:
            for day in range((min(366, len(df)))):
                for column in ["prcp", "wdir", "wspd", "pres"]:
                    if pd.isna(df.at[day, column]):
                        df.at[day, column] = median_by_day.at[day + start_offset, column]

    def clip_snow(self) -> None:
        """
        Clip the snow values to the max of 800.
        """
        for df in self.dataframes:
            df["snow"] = df["snow"].clip(upper=800)


class DataCleanerAndSaver(DataCleaner):
    def __init__(self, dataframes: list[pd.DataFrame], data_paths: list[Path]):
        super().__init__(dataframes)
        self.data_paths = data_paths

    def save_data(self) -> None:
        """
        Save the cleaned dataframes to csv files.
        """
        for df, raw_path in zip(self.dataframes, self.data_paths):
            processed_file_path = str(raw_path).replace("raw", "processed")
            prepare_directory(Path(processed_file_path).parent)
            df.to_csv(processed_file_path, index=False)


class DataCleanerFromDict(DataCleaner):
    def __init__(self, df: pd.DataFrame, city_name: str, date_offset: int):
        super().__init__([df], ["tsun", "wpgt"])
        self.city_name = city_name
        self.date_offset = date_offset

    def get_cleaned_df(self) -> None:
        self.drop_columns()
        self.handle_NaN_based_on_trend()
        median_file = Path(MEDIAN_DIR) / f"{self.city_name}.csv"
        self.handle_NaN_based_on_sesonal_pattern(median_file, self.date_offset)
        self.clip_snow()
