import mlflow
from mlflow.models.signature import infer_signature
from functools import wraps
import numpy as np


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
        with mlflow.start_run():  # We can set run name here
            mlflow.log_param("Learning Rate", optimizer.param_groups[0]["lr"])
            # add more parameters here when we establish experiments

            results = func(*args, **kwargs)

            for epoch in range(epochs):
                mlflow.log_metric("Train_MSE", results["Train_MSE"][epoch], step=epoch)
                mlflow.log_metric("Test_MSE", results["Test_MSE"][epoch], step=epoch)
                mlflow.log_metric("Train_MAE", results["Train_MAE"][epoch], step=epoch)
                mlflow.log_metric("Test_MAE", results["Test_MAE"][epoch], step=epoch)
                mlflow.log_metric("Train_RMSE", results["Train_RMSE"][epoch], step=epoch)
                mlflow.log_metric("Test_RMSE", results["Test_RMSE"][epoch], step=epoch)

            sample_input = np.expand_dims(train_dataloader.dataset[0][0].numpy(), axis=0)
            sample_output = model(train_dataloader.dataset[0][0].to(device)).detach().cpu().numpy()

            signature = infer_signature(sample_input, sample_output)
            mlflow.pytorch.log_model(model, "models", signature=signature)

        return results

    return wrapper
