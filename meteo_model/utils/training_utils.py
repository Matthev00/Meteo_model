import mlflow
from mlflow.models.signature import infer_signature
from functools import wraps
from meteo_model.model.weather_model_lstm import WeatherModelLSTM
from meteo_model.model.weather_model_tcn import WeatherModelTCN


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
        with mlflow.start_run():
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
                mlflow.log_param("Model Type", "TCN")

            results = func(*args, **kwargs)

            for epoch in range(epochs):
                mlflow.log_metric("Train_MSE", results["Train_MSE"][epoch], step=epoch)
                mlflow.log_metric("Test_MSE", results["Test_MSE"][epoch], step=epoch)
                mlflow.log_metric("Train_MAE", results["Train_MAE"][epoch], step=epoch)
                mlflow.log_metric("Test_MAE", results["Test_MAE"][epoch], step=epoch)
                mlflow.log_metric("Train_RMSE", results["Train_RMSE"][epoch], step=epoch)
                mlflow.log_metric("Test_RMSE", results["Test_RMSE"][epoch], step=epoch)

            sample_input = train_dataloader.dataset[0][0].numpy()
            sample_output = model(train_dataloader.dataset[0][0].to(device)).detach().cpu().numpy()

            signature = infer_signature(sample_input, sample_output)
            mlflow.pytorch.log_model(model, "models", signature=signature)

        return results

    return wrapper
