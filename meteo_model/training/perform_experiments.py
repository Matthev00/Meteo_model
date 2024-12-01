import os
import torch
import optuna
from meteo_model.training.engine import train
from meteo_model.data.data_loader import create_dataloaders
from meteo_model.model.weather_model_lstm import WeatherModelLSTM
from meteo_model.model.weather_model_tcn import WeatherModelTCN
from meteo_model.training.config import OPTUNA_STORAGE_PATH


def objective_lstm(trial):
    batch_size = trial.suggest_int("batch_size", 2, 32, step=2)
    lr = trial.suggest_loguniform("lr", 1e-5, 1e-1)
    epochs = trial.suggest_int("epochs", 5, 50)
    num_layers = trial.suggest_int("num_layers", 1, 8)
    hidden_size = trial.suggest_int("hidden_size", 16, 256)
    input_len = trial.suggest_int("input_len", 4, 32)
    location = trial.suggest_categorical(
        "location", [["BIALYSTOK", "WARSAW", "WROCLAW", "KRAKOW", "POZNAN"], ["WARSAW"]]
    )
    output_len = 8
    split_ratio = 0.8
    device = "cuda" if torch.cuda.is_available() else "cpu"
    num_features = 8

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
        num_features=num_features,
        num_locations=len(location),
        output_len=output_len,
        hidden_size=hidden_size,
        num_layers=num_layers,
    )
    model = model.to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)

    results = train(
        model=model,
        train_dataloader=train_loader,
        test_dataloader=test_loader,
        optimizer=optimizer,
        loss_fn=loss_fn,
        epochs=epochs,
        device=device,
        enable_logging=True,
        experiment_name="LSTM_first_training",
    )

    return results["Test_MSE"][-1]


def objective_tcn(trial):
    batch_size = trial.suggest_int("batch_size", 2, 32, step=2)
    lr = trial.suggest_loguniform("lr", 1e-5, 1e-1)
    epochs = trial.suggest_int("epochs", 5, 50)

    kernel_size = 2
    dropout = trial.suggest_loguniform("lr", 1e-4, 2e-1)
    input_len = trial.suggest_int("input_len", 4, 32)
    location = trial.suggest_categorical(
        "location", [["BIALYSTOK", "WARSAW", "WROCLAW", "KRAKOW", "POZNAN"], ["WARSAW"]]
    )
    output_len = 8
    split_ratio = 0.8
    device = "cuda" if torch.cuda.is_available() else "cpu"
    num_features = 8

    num_channels = trial.suggest_categorical("num_channels", [[4, 2, 1, 2, 4], [5, 5], [8]])
    num_channels.append(num_features)

    train_loader, test_loader = create_dataloaders(
        location=location,
        input_len=input_len,
        output_len=output_len,
        split_ratio=split_ratio,
        batch_size=batch_size,
        num_workers=os.cpu_count(),
    )

    loss_fn = torch.nn.MSELoss()
    model = WeatherModelTCN(
        num_features=num_features,
        num_locations=len(location),
        output_len=output_len,
        num_channels=num_channels,
        kernel_size=kernel_size,
        dropout=dropout,
    )
    model = model.to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)

    results = train(
        model=model,
        train_dataloader=train_loader,
        test_dataloader=test_loader,
        optimizer=optimizer,
        loss_fn=loss_fn,
        epochs=epochs,
        device=device,
        enable_logging=True,
        experiment_name="TCN_first_training",
    )

    return results["Test_MSE"][-1]


def create_study_for_(objective, name):
    study = optuna.create_study(
        study_name=name,
        direction="minimize",
        storage=OPTUNA_STORAGE_PATH,
        load_if_exists=True,
    )
    study.optimize(objective, n_trials=1)

    print("Best trial:")
    print(f"  Value: {study.best_trial.value}")
    print(f"  Params: {study.best_trial.params}")


def main():
    create_study_for_(objective_lstm, "weather_model_LSTM")
    create_study_for_(objective_tcn, "weather_model_TCN")


if __name__ == "__main__":
    main()
