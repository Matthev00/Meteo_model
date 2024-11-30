from meteo_model.utils.evaluation_utils import visualize_predictions
from meteo_model.utils.model_utils import load_model
from meteo_model.data.data_loader import create_dataloaders
from pathlib import Path
import os
import torch
from meteo_model.data.normaliser import inverse_normalize_data
import pandas as pd
import json
from meteo_model.data.config import PATH_TO_STATS, NORM_COLUMNS, COLUMNS


def prepare_df(
    X: torch.Tensor, y: torch.Tensor, pred: torch.Tensor
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    with open(PATH_TO_STATS) as f:
        stats_ = json.load(f)

    X_df = pd.DataFrame(X[0][0], columns=NORM_COLUMNS)
    y_df = pd.DataFrame(y[0][0], columns=NORM_COLUMNS)
    pred_df = pd.DataFrame(pred[0], columns=NORM_COLUMNS)

    X_df = inverse_normalize_data(X_df, stats_)
    y_df = inverse_normalize_data(y_df, stats_)
    pred_df = inverse_normalize_data(pred_df, stats_)

    return X_df, y_df, pred_df


def main():
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    num_workers = os.cpu_count() or 1

    _, test_dl = create_dataloaders(
        root_dir=Path("data/normalized"),
        location=["WARSAW"],
        input_len=32,
        output_len=4,
        split_ratio=0.8,
        batch_size=16,
        num_workers=num_workers,
    )
    X, y = next(iter(test_dl))

    model = load_model("Mete-test", 4)
    model.eval()
    with torch.inference_mode():
        pred = model(X.to(device))[0].detach().cpu().numpy()

    visualize_predictions(*prepare_df(X, y, pred), COLUMNS)


if __name__ == "__main__":
    main()
