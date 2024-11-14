# from meteo_model.data.data_loader import create_dataloaders
from meteo_model.training.engine import train
import os
import torch
import itertools
from torch.utils.data import DataLoader, TensorDataset


class Model(torch.nn.Module):
    def __init__(self, input_size, output_size):
        super(Model, self).__init__()
        self.linear = torch.nn.Linear(input_size, output_size)

    def forward(self, x):
        return self.linear(x)
    
def create_dataloaders(input_len, output_len, batch_size, location, split_ratio, num_workers):
    num_samples = 100 
    inputs = torch.randn(num_samples, input_len)
    targets = torch.randn(num_samples, output_len)

    dataset = TensorDataset(inputs, targets)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

    return dataloader, dataloader


def main():
    NUM_WORKERS = os.cpu_count()
    OUTPUT_LEN = 8
    SPLIT_RATIO = 0.8
    DEVICE = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    params = {
        "location": [["BIALYSTOK", "WARSAW", "WROCLAW", "KRAKOW", "POZNAN"], ["BIALYSTOK"]],
        "input_len": [32, 16],
        "batch_size": [16, 8],
        "lr": [0.001, 0.01],
        "epochs": [5, 10],
    }

    keys, values = zip(*params.items())
    combinations = [dict(zip(keys, v)) for v in itertools.product(*values)]

    for combo in combinations:
        model = Model(input_size=combo["input_len"], output_size=OUTPUT_LEN).to(DEVICE)
        epochs = combo["epochs"]
        OPTIMIZER = torch.optim.Adam(model.parameters(), lr=combo["lr"])
        train_loader, test_loader = create_dataloaders(
            location=combo["location"],
            input_len=combo["input_len"],
            output_len=OUTPUT_LEN,
            split_ratio=SPLIT_RATIO,
            batch_size=combo["batch_size"],
            num_workers=NUM_WORKERS,
        )

        results = train(
            model=model,
            train_dataloader=train_loader,
            test_dataloader=test_loader,
            optimizer=torch.optim.Adam(model.parameters(), lr=0.001),
            loss_fn=torch.nn.MSELoss(),
            epochs=combo["epochs"],
            device=DEVICE,
        )


if __name__ == "__main__":
    main()
