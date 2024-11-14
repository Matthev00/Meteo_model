from meteo_model.data.data_loader import create_dataloaders
from meteo_model.training
import os


def main():
    NUM_WORKERS = os.cpu_count()
    # Get dataloaders
    train_loader, val_loader = create_dataloaders(
        location=["BIALYSTOK", "WARSAW", "WROCLAW", "KRAKOW", "POZNAN"],
        input_len=32,
        output_len=8,
        split_ratio=0.8,
        batch_size=16,
        num_workers=
    )

    # Train model
    train(train_loader, val_loader)