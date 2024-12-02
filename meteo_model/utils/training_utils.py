import os
import mlflow
import torch
from mlflow.models.signature import infer_signature
from functools import wraps
import argparse
from meteo_model.model.weather_model_lstm import WeatherModelLSTM
from meteo_model.model.weather_model_tcn import WeatherModelTCN
from meteo_model.data.config import LOCATIONS_NAMES


def mlflow_logging(func):
    @wraps(func)
    def wrapper(*args, **kwargs) -> dict[str, list[float]]:
        enable_logging = kwargs.pop("enable_logging", True)
        experiment_name = kwargs.pop("experiment_name", "MeteoModelForecasting")
        if not enable_logging:
            return func(*args, **kwargs)

        model = kwargs.get("model")
        train_dataloader = kwargs.get("train_dataloader")
        optimizer = kwargs.get("optimizer")
        epochs = kwargs.get("epochs")
        device = kwargs.get("device", "cuda")

        mlflow.set_experiment(experiment_name)
        with mlflow.start_run() as run:
            mlflow.log_param("Learning Rate", optimizer.param_groups[0]["lr"])
            mlflow.log_param("Batch Size", train_dataloader.batch_size)
            mlflow.log_param("Input Length", train_dataloader.dataset.dataset.input_len)
            mlflow.log_param("Epochs", epochs)
            mlflow.log_param("Locations_size", len(train_dataloader.dataset.dataset.location))

            if isinstance(model, WeatherModelLSTM):
                mlflow.log_param("Hidden Size", model.hidden_size)
                mlflow.log_param("Number of Layers", model.num_layers)
                mlflow.log_param("Model Type", "LSTM")
            if isinstance(model, WeatherModelTCN):
                mlflow.log_param("Kernel Size", model.kernel_size)
                mlflow.log_param("Dropout", model.dropout)
                mlflow.log_param("Number of Channels", model.num_channels)
                mlflow.log_param("Model Type", "TCN")

            results = func(*args, **kwargs)

            for epoch in range(epochs):
                mlflow.log_metric("Train_MSE", results["Train_MSE"][epoch], step=epoch)
                mlflow.log_metric("Test_MSE", results["Test_MSE"][epoch], step=epoch)
                mlflow.log_metric("Train_MAE", results["Train_MAE"][epoch], step=epoch)
                mlflow.log_metric("Test_MAE", results["Test_MAE"][epoch], step=epoch)
                mlflow.log_metric("Train_RMSE", results["Train_RMSE"][epoch], step=epoch)
                mlflow.log_metric("Test_RMSE", results["Test_RMSE"][epoch], step=epoch)

            

            if isinstance(model, WeatherModelLSTM):
                sample_input = train_dataloader.dataset[0][0].numpy()
                sample_output = model(train_dataloader.dataset[0][0].to(device)).detach().cpu().numpy()
                signature = infer_signature(sample_input, sample_output)
                mlflow.pytorch.log_model(model, "models", signature=signature)
            if isinstance(model, WeatherModelTCN):
                tcn_model_dir_path = run.info.artifact_uri + "/model_dict"
                if tcn_model_dir_path.startswith("file://"):
                    tcn_model_dir_path = tcn_model_dir_path[6:]
                os.makedirs(tcn_model_dir_path, exist_ok=True)
                torch.save(model.state_dict(), tcn_model_dir_path + "/model_dict.pth")
        return results

    return wrapper


def str2bool(v):
    if isinstance(v, bool):
        return v
    if v.lower() in ("yes", "true", "t", "y", "1"):
        return True
    elif v.lower() in ("no", "false", "f", "n", "0"):
        return False
    else:
        raise argparse.ArgumentTypeError("Boolean value expected.")


def parse_arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Script train METEO MODEL.")

    parser.add_argument(
        "--n_locations", type=int, default=1, help="Number of locations to train the model"
    )
    parser.add_argument("--input_len", type=int, default=32, help="Number of input time steps")
    parser.add_argument("--output_len", type=int, default=4, help="Number of output time steps")
    parser.add_argument(
        "--batch_size", type=int, default=16, help="Batch size for training and testing"
    )

    parser.add_argument(
        "--model_type", type=str, default="lstm", help="Type of model to train (lstm or tcn)"
    )
    parser.add_argument(
        "--hidden_size", type=int, default=128, help="Hidden size for the LSTM model"
    )
    parser.add_argument(
        "--num_layers", type=int, default=2, help="Number of layers for the LSTM model"
    )
    parser.add_argument(
        "--kernel_size", type=int, default=2, help="Kernel size of TCN"
    )
    parser.add_argument(
        "--dropout", type=int, default=0.1, help="Dropout for TCN"
    )
    parser.add_argument(
        "--num_channels",
        type=int,
        nargs="+",  
        help="Number of output channels in TCN. It also determines the number of layers of that TCN",
    )

    parser.add_argument("--lr", type=float, default=0.001, help="Learning rate for the optimizer")
    parser.add_argument("--epochs", type=int, default=5, help="Number of epochs for training")
    parser.add_argument(
        "--enable_logging", type=str2bool, default=True, help="Enable logging with MLFlow"
    )
    parser.add_argument(
        "--experiment_name",
        type=str,
        default="Default_experiment",
        help="Name of the MLFlow experiment",
    )

    args = parser.parse_args()

    args.location = LOCATIONS_NAMES[: args.n_locations]
    return args
