import os
import glob
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import json
from pathlib import Path
from config import PATH_TO_STATS, PATHS_TO_DATA_FILES_STR


def get_dataframe(paths):
    dataframes = [pd.read_csv(file_name) for file_name in paths]
    return pd.concat(dataframes)

def get_stat_json(df : pd.DataFrame, indent = 4) -> str:
    return df.describe().to_json(indent=indent)


def create_stat_file():
    PATHS_TO_DATA_FILES= glob.glob(PATHS_TO_DATA_FILES_STR)
    with open(PATH_TO_STATS, 'w') as f:
        df = get_dataframe(PATHS_TO_DATA_FILES)
        stats_json = get_stat_json(df)
        f.writelines(stats_json)



if __name__ == "__main__":
    create_stat_file()
    
    