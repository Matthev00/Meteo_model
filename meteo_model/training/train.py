from meteo_model.data.data_loader import create_dataloaders
from meteo_model.training.engine import train
from meteo_model.model.weather_model_lstm import WeatherModelLSTM
from meteo_model.model.weather_model_tcn import WeatherModelTCN
from meteo_model.utils.training_utils import parse_arguments
import torch
from pathlib import Path
import os


def main():
    args = parse_arguments()
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    num_workers = os.cpu_count() or 1

    train_dl, test_dl = create_dataloaders(
        root_dir=Path("data/normalized"),
        location=args.location,
        input_len=args.input_len,
        output_len=args.output_len,
        split_ratio=0.8,
        batch_size=args.batch_size,
        num_workers=num_workers,
    )

    if args.model_type == "lstm":
        model = WeatherModelLSTM(
            num_features=9,
            num_locations=args.n_locations,
            output_len=args.output_len,
            hidden_size=args.hidden_size,
            num_layers=args.num_layers,
        ).to(device)
    if args.model_type == "tcn":
        model = WeatherModelTCN(
            num_features=9,
            num_locations=args.n_locations,
            output_len=args.output_len,
            num_channels=args.num_channels,
            kernel_size=args.kernel_size,
            dropout=args.dropout,
        ).to(device)

    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    loss_fn = torch.nn.MSELoss()

    train(
        model=model,
        train_dataloader=train_dl,
        test_dataloader=test_dl,
        optimizer=optimizer,
        loss_fn=loss_fn,
        epochs=args.epochs,
        device=device,
        enable_logging=args.enable_logging,
        experiment_name=args.experiment_name,
    )


if __name__ == "__main__":
    main()
