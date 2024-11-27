from meteo_model.utils.evaluation_utils import visualize_predictions
from meteo_model.utils.model_utils import load_model
from meteo_model.data.data_loader import create_dataloaders
from pathlib import Path
import os
import torch


def main():
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    num_workers = os.cpu_count() or 1
    feature_names = [
        "Avg Temperature",
        "Min Temperature",
        "Max Temperature",
        "Percepitation",
        "Snow",
        "Wind Direction",
        "Wind Speed",
        "Pressure",
    ]

    _, test_dl = create_dataloaders(
        root_dir=Path("data/processed/weather_data"),
        location=["BIALYSTOK", "WARSAW", "WROCLAW", "KRAKOW", "POZNAN"],
        input_len=32,
        output_len=8,
        split_ratio=0.8,
        batch_size=16,
        num_workers=num_workers,
    )

    sample = next(iter(test_dl))
    X, y = sample
    model = load_model("Mete-test", 2)
    model.eval()
    pred = model(X.to(device))[0].detach().cpu().numpy()
    visualize_predictions(X[0], y[0], pred, test_dl.dataset.features_names)


if __name__ == "__main__":
    main()
