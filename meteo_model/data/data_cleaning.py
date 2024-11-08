import numpy as np
import pandas as pd 
from pathlib import Path

from meteo_model.utils.file_utils import get_station_name_from_city_name, prepare_directory


def get_raw_data(station: str, data_dir: Path = Path("data/raw/weather_data"), year_range: tuple[int, int] = (2012, 2024)) -> list[pd.DataFrame]:
    """
    Load data from csv files and return a list of dataframes.
    """
    data = []
    data_paths = []
    for year in range(year_range[0], year_range[1] + 1):
        file_path = data_dir / str(year) / f"{station}_weather_data.csv"
        if file_path.exists():
            data.append(pd.read_csv(file_path))
            data_paths.append(file_path)
    return data, data_paths


def drop_columns(dataframes: list[pd.DataFrame]) -> None:
    """
    Drop unnecessary columns from the dataframes in place.
    ['station', 'tsun', 'wpgt']
    """
    for df in dataframes:
        df.drop(columns=['station', 'tsun', 'wpgt'], inplace=True)


def handle_NaN_based_on_trend(dataframes: list[pd.DataFrame]) -> None:
    """
    Handle missing values in the data based on trend. In Place.
    """
    for df in dataframes:
        df["tavg"] = df["tavg"].interpolate(method='linear', limit=2, limit_direction='both')
        df["tmin"] = df["tmin"].interpolate(method='linear', limit=2, limit_direction='both')
        df["tmax"] = df["tmax"].interpolate(method='linear', limit=2, limit_direction='both')
        df['prcp'] = df['prcp'].interpolate(method='nearest', limit=2, limit_direction='both')
        df['snow'] = df['snow'].fillna(0)
        df['pres'] = df['pres'].interpolate(method='linear', limit=2, limit_direction='both')
        df['wdir'] = df['wdir'].interpolate(method='nearest', limit=3, limit_direction='both')
        df['wspd'] = df['wspd'].interpolate(method='linear', limit=2, limit_direction='both')


def calculate_median_by_day(dataframes: list[pd.DataFrame]) -> None:
    """
    Calculate the median values for each day of the year.
    """
    median_by_day = pd.DataFrame(index=range(366), columns=['prcp', 'wdir', 'wspd', 'pres'])
    
    for day in range(366):  
        day_data = [df.iloc[day] for df in dataframes if len(df) > day] 
        day_df = pd.DataFrame(day_data).dropna(how='all')
        
        if not day_df.empty:
            median_by_day.loc[day, 'prcp'] = day_df['prcp'].median() if 'prcp' in day_df else np.nan
            median_by_day.loc[day, 'wdir'] = day_df['wdir'].median() if 'wdir' in day_df else np.nan
            median_by_day.loc[day, 'wspd'] = day_df['wspd'].median() if 'wspd' in day_df else np.nan
            median_by_day.loc[day, 'pres'] = day_df['pres'].median() if 'pres' in day_df else np.nan

    return median_by_day


def handle_NaN_based_on_sesonal_pattern(dataframes: list[pd.DataFrame]) -> None:
    """
    Handle missing values in the data based on group. In Place.
    """
    median_by_day = calculate_median_by_day(dataframes)
    for df in dataframes:
        for day in range((min(366, len(df)))):
            for column in ['prcp', 'wdir', 'wspd', 'pres']:
                if pd.isna(df.at[day, column]):
                    df.at[day, column] = median_by_day.at[day, column]


def clip_snow(dataframes: list[pd.DataFrame]) -> None:
    """
    Clip the snow values to the max of 800.
    """
    for df in dataframes:
        df['snow'] = df['snow'].clip(upper=800)


def save_data(dataframes: list[pd.DataFrame], data_paths: list[Path]) -> None:
    """
    Save the cleaned dataframes to csv files.
    """
    for df, raw_path in zip(dataframes, data_paths):
        processed_file_path = str(raw_path).replace("raw", "processed")
        prepare_directory(Path(processed_file_path).parent)
        df.to_csv(processed_file_path, index=False)


def clean_and_save_data(city_name: str) -> None:
    station = get_station_name_from_city_name(city_name)
    data, data_paths = get_raw_data(station)
    yield f"{city_name}: {len(data)} files found."
    drop_columns(data)
    yield f"{city_name}: Columns dropped."
    handle_NaN_based_on_trend(data)
    yield f"{city_name}: NaN handled based on trend."
    handle_NaN_based_on_sesonal_pattern(data)
    yield f"{city_name}: NaN handled based on seasonal pattern."
    clip_snow(data)
    yield f"{city_name}: Snow values clipped."
    save_data(data, data_paths)
    yield f"{city_name}: Data saved."


def main():
    cities = ['Bialystok', 'Warszawa', 'Wroclaw', 'Krakow', 'Poznan']
    for city in cities:
        for message in clean_and_save_data(city):
            print(message)
        print(40 * '-')


if __name__ == "__main__":
    main()