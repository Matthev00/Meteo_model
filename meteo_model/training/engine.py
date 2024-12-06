import torch
import math
from tqdm.auto import tqdm
from meteo_model.utils.training_utils import mlflow_logging


@mlflow_logging
def train(
    model: torch.nn.Module,
    train_dataloader: torch.utils.data.DataLoader,
    test_dataloader: torch.utils.data.DataLoader,
    optimizer: torch.optim.Optimizer,
    loss_fn: torch.nn.Module,
    epochs: int,
    device: str = "cuda",
    enable_logging: bool = True,
    experiment_name: str = "MeteoModelForecasting",
    scheduler: torch.optim.lr_scheduler = None,
) -> dict[str, list[float]]:

    results: dict[str, list[float]] = {
        "Train_MSE": [],
        "Test_MSE": [],
        "Train_MAE": [],
        "Test_MAE": [],
        "Train_RMSE": [],
        "Test_RMSE": [],
    }
    for epoch in tqdm(range(epochs)):
        train_mse, train_mae, train_rmse = train_step(
            model=model,
            train_dataloader=train_dataloader,
            optimizer=optimizer,
            loss_fn=loss_fn,
            device=device,
        )
        test_mse, test_mae, test_rmse = test_step(
            model=model, test_dataloader=test_dataloader, loss_fn=loss_fn, device=device
        )
        results["Train_MSE"].append(train_mse)
        results["Test_MSE"].append(test_mse)
        results["Train_MAE"].append(train_mae)
        results["Test_MAE"].append(test_mae)
        results["Train_RMSE"].append(train_rmse)
        results["Test_RMSE"].append(test_rmse)

        if scheduler:
            scheduler.step(test_mse)

    return results


def train_step(
    model: torch.nn.Module,
    train_dataloader: torch.utils.data.DataLoader,
    optimizer: torch.optim.Optimizer,
    loss_fn: torch.nn.Module,
    device="cuda",
) -> tuple[float, float, float]:
    model.train()
    total_loss = 0
    total_mae = 0.0
    total_samples = 0
    for batch, (inputs, targets) in enumerate(train_dataloader):
        inputs, targets = inputs.to(device), targets.to(device)
        optimizer.zero_grad()
        outputs = model(inputs)
        loss = loss_fn(outputs, targets)
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
        total_mae += torch.abs(outputs - targets).sum().item()
        total_samples += targets.size(0) * targets.size(2)
    mse = total_loss / total_samples
    mae = total_mae / total_samples
    rmse = math.sqrt(mse)
    return mse, mae, rmse


def test_step(
    model: torch.nn.Module,
    test_dataloader: torch.utils.data.DataLoader,
    loss_fn: torch.nn.Module,
    device="cuda",
) -> tuple[float, float, float]:
    model.eval()
    total_loss = 0
    total_mae = 0.0
    total_samples = 0
    with torch.inference_mode():
        for batch, (inputs, targets) in enumerate(test_dataloader):
            inputs, targets = inputs.to(device), targets.to(device)
            outputs = model(inputs)
            loss = loss_fn(outputs, targets)
            total_loss += loss.item()
            total_mae += torch.abs(outputs - targets).sum().item()
            total_samples += targets.size(0) * targets.size(2)
    mse = total_loss / total_samples
    mae = total_mae / total_samples
    rmse = math.sqrt(mse)
    return mse, mae, rmse
