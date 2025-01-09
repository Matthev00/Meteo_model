import glob
import pandas as pd
from pathlib import Path
from meteo_model.data.config import PATH_TO_STATS, PATHS_TO_DATA_FILES_STR
from typing import Sequence


def get_dataframe(paths: Sequence[str | Path]) -> pd.DataFrame:
    dataframes = [pd.read_csv(file_name) for file_name in paths]
    return pd.concat(dataframes)


def get_stat_json(df: pd.DataFrame, indent: int = 4) -> str:
    return df.describe().to_json(indent=indent)


def create_stat_file() -> None:
    PATHS_TO_DATA_FILES = glob.glob(PATHS_TO_DATA_FILES_STR)
    with open(PATH_TO_STATS, "w") as f:
        df = get_dataframe(PATHS_TO_DATA_FILES)
        stats_json = get_stat_json(df)
        f.writelines(stats_json)


if __name__ == "__main__":
    create_stat_file()
