import os
import torch
import optuna
import argparse
from meteo_model.training.engine import train
from meteo_model.data.data_loader import create_dataloaders
from meteo_model.model.weather_model_lstm import WeatherModelLSTM
from meteo_model.training.config import OPTUNA_STORAGE_PATH_LSTM


def objective_lstm(trial, experiment_name, n_days):
    batch_size = trial.suggest_int("batch_size", 2, 32, step=2)
    lr = trial.suggest_float("lr", 1e-5, 1e-1, log=True)
    epochs = trial.suggest_int("epochs", 5, 30)
    num_layers = trial.suggest_int("num_layers", 1, 8)
    hidden_size = trial.suggest_int("hidden_size", 16, 256)
    input_len = trial.suggest_int("input_len", 4, 32)
    location = trial.suggest_categorical(
        "location", ("BIALYSTOK, WARSAW, WROCLAW, KRAKOW, POZNAN", "WARSAW")
    )
    output_len = n_days
    split_ratio = 0.8
    device = "cuda" if torch.cuda.is_available() else "cpu"
    num_features = 9
    location = location.split(", ")

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
        experiment_name=experiment_name,
    )

    return results["Test_MSE"][-1]


def create_study_for_(objective, name, n_days):
    study = optuna.create_study(
        study_name=name,
        direction="minimize",
        storage=OPTUNA_STORAGE_PATH_LSTM,
        load_if_exists=True,
    )
    study.optimize(lambda trial: objective(trial, name, n_days), n_trials=60)

    print("Best trial:")
    print(f"  Value: {study.best_trial.value}")
    print(f"  Params: {study.best_trial.params}")


def main():
    parser = argparse.ArgumentParser(description="Perform LSTM experiments")
    parser.add_argument("--n_days", type=int, help="Number of days for prediction")
    parser.add_argument("--experiment_name", type=str, help="Name of the experiment")
    args = parser.parse_args()

    create_study_for_(objective_lstm, args.experiment_name, args.n_days)


if __name__ == "__main__":
    main()
