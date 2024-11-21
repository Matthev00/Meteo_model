import os
import torch
import optuna
from meteo_model.training.engine import train
from meteo_model.data.data_loader import create_dataloaders
from meteo_model.model.weather_model_lstm import WeatherModelLSTM


def objective(trial):
    input_len = trial.suggest_int("input_len", 1, 14)
    batch_size = trial.suggest_int("batch_size", 1, 64)
    lr = trial.suggest_float("lr", 1e-5, 1e-1)
    epochs = trial.suggest_int("epochs", 1, 10)
    hidden_size = trial.suggest_int("hidden_size", 1, 64)
    num_layers = trial.suggest_int("num_layers", 1, 4)

    output_len = 8
    split_ratio = 0.8
    location = ["BIALYSTOK", "WARSAW", "WROCLAW", "KRAKOW", "POZNAN"]

    train_loader, test_loader = create_dataloaders(
        location=location,
        input_len=input_len,
        output_len=output_len,
        split_ratio=split_ratio,
        batch_size=batch_size,
        num_workers=os.cpu_count(),
    )

    loss_fn = torch.nn.MSELoss()
    model = WeatherModelLSTM(
        num_features=8,
        num_locations=len(location),
        output_len=output_len,
        hidden_size=hidden_size,
        num_layers=num_layers,
    )
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)

    results = train(
        model=model,
        train_dataloader=train_loader,
        test_dataloader=test_loader,
        optimizer=optimizer,
        loss_fn=loss_fn,
        epochs=epochs,
        device="cpu",
        enable_logging=False,
    )

    return results["Test_MSE"][-1]


def main():
    study = optuna.create_study(direction="minimize")
    study.optimize(objective, n_trials=10)

    print("Best trial:")
    print(f"  Value: {study.best_trial.value}")
    print(f"  Params: {study.best_trial.params}")


if __name__ == "__main__":
    main()
