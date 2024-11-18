"""Module for normalising weather attributes"""
import json
import numpy as np
from pathlib import Path
from config import PATH_TO_STATS, PATHS_TO_DATA_FILES_STR
from get_stats import create_stat_file
import pandas as pd


if not Path(PATH_TO_STATS).exists():
    create_stat_file()

with open(PATH_TO_STATS) as f:
    stats = json.load(f)


def _normalise_norm_like(array, attr_name: str):
    mean = stats[attr_name]["mean"]
    std = stats[attr_name]["std"]
    return (array - mean)/std



def _normalise_tavg(array):
    return _normalise_norm_like(array, "tavg")

def _normalise_tmin(array):
    return _normalise_norm_like(array, "tmin")

def _normalise_tmax(array):
    return _normalise_norm_like(array, "tmax")

def _normalise_prcp(array):
    return np.log1p(array)

def _normalise_snow(array):
    mean_snow = stats["snow"]["mean"]
    mean_snow_log = np.log1p(mean_snow)
    return np.log1p(array) / mean_snow

def _normalise_wdir(array):
    rad_array = np.deg2rad(array)
    return np.sin(rad_array), np.cos(rad_array)

def _normalise_wspd(array):
    return _normalise_norm_like(array, "wspd")

def _normalise_pres(array):
    return _normalise_norm_like(array, "pres")

def normalize_data(df):
    columns = ['tavg', 'tmin', 'tmax', 'prcp', 'snow', 'sin_wdir', 'cos_wdir', 'wspd', 'wpgt', 'pres',
       'tsun']
    normalised_df = pd.DataFrame(columns=columns)
    normalised_df["tavg"] = _normalise_tavg(df["tavg"])
    normalised_df["tmin"] = _normalise_tmin(df["tmin"])
    normalised_df["tmax"] = _normalise_tmax(df["tmax"])
    normalised_df["prcp"] = _normalise_prcp(df["prcp"])
    normalised_df["snow"] = _normalise_snow(df["snow"])
    sin_wdir, cos_wdir = _normalise_wdir(df["wdir"])
    normalised_df["sin_wdir"] = sin_wdir
    normalised_df["cos_wdir"] = cos_wdir
    normalised_df["wspd"] = _normalise_wspd(df["wspd"])
    normalised_df["pres"] = _normalise_pres(df["pres"])
    return normalised_df



if __name__== "__main__":
    from get_stats import get_dataframe
    import matplotlib.pyplot as plt
    from glob import glob

    variables = ['tavg', 'tmin', 'tmax', 'prcp', 'snow', 'sin_wdir', 'cos_wdir', 'wspd', 'pres']
    plt.figure(figsize=(12, 10))
    huge_df = get_dataframe(glob(PATHS_TO_DATA_FILES_STR))
    normalised = normalize_data(huge_df)

    for i, var in enumerate(variables, 1):
        plt.subplot(5, 2, i)  # 5 rows, 2 columns grid
        plt.hist(normalised.dropna(subset=[var])[var], bins=100, color='blue')
        plt.title(var)
        plt.xlabel('Value')
        plt.ylabel('Frequency')

    plt.tight_layout()
    plt.savefig("reports/figures/data/data_featues_histograms_normalised.png")
    plt.show()
