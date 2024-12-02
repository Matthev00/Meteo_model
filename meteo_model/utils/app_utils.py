import torch
import pandas as pd
import json
from meteo_model.data.normaliser import inverse_normalize_data
from meteo_model.data.config import PATH_TO_STATS, NORM_COLUMNS
from meteo_model.utils.model_utils import load_model
from meteo_model.data.api.api_data_provider import get_weather_tensor_for_days
from datetime import datetime, timedelta


def prepare_pred_df(pred: torch.Tensor) -> pd.DataFrame:
    with open(PATH_TO_STATS) as f:
        stats_ = json.load(f)

    pred_df = pd.DataFrame(pred[0], columns=NORM_COLUMNS)
    pred_df = inverse_normalize_data(pred_df, stats_)

    return pred_df


def predict(n_days: int) -> pd.DataFrame:
    X = get_weather_tensor_for_days(n_days, ["WARSAW"])  # Change location here
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    # Select best model model
    model = load_model("Mete-test", 4)
    model.eval()
    with torch.inference_mode():
        pred = model(X.to(device))[0].detach().cpu().numpy()

    return prepare_pred_df(pred).iloc[:n_days, :]


def get_dates(n_days: int) -> pd.DatetimeIndex:
    start_date = datetime.now() + timedelta(days=1)
    return pd.date_range(start=start_date, periods=n_days).strftime("%d-%m-%Y")
