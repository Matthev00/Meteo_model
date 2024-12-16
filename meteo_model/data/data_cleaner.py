import pandas as pd
import numpy as np
from pathlib import Path

from meteo_model.utils.file_utils import prepare_directory


class DataCleaner:
    def __init__(self, dataframes, columns_to_drop = ["station", "tsun", "wpgt"]):
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

    def handle_NaN_based_on_sesonal_pattern(self) -> None:
        """
        Handle missing values in the data based on group. In Place.
        """
        median_by_day = self.calculate_median_by_day()
        for df in self.dataframes:
            for day in range((min(366, len(df)))):
                for column in ["prcp", "wdir", "wspd", "pres"]:
                    if pd.isna(df.at[day, column]):
                        df.at[day, column] = median_by_day.at[day, column]

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
    def __init__(self, dataframes_dict: dict[str, pd.DataFrame]):
        super().__init__(dataframes_dict.values(), ["tsun", "wpgt"])
        self.dataframes_dict = dataframes_dict

    def get_cleaned_dataframes_dict(self):
        self.drop_columns()
        self.handle_NaN_based_on_trend()
        self.handle_NaN_based_on_sesonal_pattern()
        self.clip_snow()
        return self.dataframes_dict