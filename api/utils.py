import torch
import pandas as pd
import json
from datetime import datetime, timedelta
from meteo_model.data.normaliser import inverse_normalize_data
from meteo_model.data.config import PATH_TO_STATS, NORM_COLUMNS


def prepare_pred_df(pred: torch.Tensor) -> pd.DataFrame:
    with open(PATH_TO_STATS) as f:
        stats_ = json.load(f)

    pred_df = pd.DataFrame(pred[0], columns=NORM_COLUMNS)
    pred_df = inverse_normalize_data(pred_df, stats_)

    return pred_df


def get_dates(pred_end_day: str, n_days: int) -> pd.DatetimeIndex:
    datetime.strptime(pred_end_day, "%Y-%m-%d")
    start_date = datetime.strptime(pred_end_day, "%Y-%m-%d") + timedelta(days=1)
    return pd.date_range(start=start_date, periods=n_days).strftime("%d-%m-%Y")
