from meteo_model.data.data_loader import create_dataloaders
from pathlib import Path
import os
import torch


def get_weather_tensor_for_days(days: int, location_names: list[str]) -> torch.Tensor:
    _, test_dl = create_dataloaders(
        root_dir=Path("data/normalized"),
        location=location_names,
        input_len=32,
        output_len=4,
        split_ratio=0.8,
        batch_size=16,
        num_workers=os.cpu_count() or 1,
    )
    X, _ = next(iter(test_dl))
    return X
