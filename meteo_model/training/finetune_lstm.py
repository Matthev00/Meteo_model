from meteo_model.data.data_loader import create_dataloaders
from meteo_model.training.engine import train
from meteo_model.utils.model_utils import load_model
from meteo_model.data.config import LOCATIONS_NAMES
import torch
from pathlib import Path
import os


def main():
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    num_workers = os.cpu_count() or 1

    train_dl, test_dl = create_dataloaders(
        root_dir=Path("data/normalized"),
        location=LOCATIONS_NAMES,
        input_len=8,
        output_len=8,
        split_ratio=0.8,
        batch_size=22,
        num_workers=num_workers,
    )

    model = load_model("LSTM-best-base", 2)

    optimizer = torch.optim.Adam(model.parameters(), lr=0.0021950336406807353)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="min", factor=0.1, patience=2, verbose=True
    )

    loss_fn = torch.nn.MSELoss()

    train(
        model=model,
        train_dataloader=train_dl,
        test_dataloader=test_dl,
        optimizer=optimizer,
        loss_fn=loss_fn,
        epochs=9,
        device=device,
        enable_logging=True,
        experiment_name="FineTuneLSTM",
        scheduler=scheduler,
    )


if __name__ == "__main__":
    main()
