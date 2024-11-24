from torch.utils.data import DataLoader, Subset
from pathlib import Path
from meteo_model.data.datasets import MeteoDataset
import os


def create_dataloaders(
    root_dir: Path = Path("data/processed/weather_data"),
    location=None,
    input_len: int = 32,
    output_len: int = 8,
    split_ratio: float = 0.8,
    batch_size: int = 16,
    num_workers: int = 1,
) -> tuple[DataLoader, DataLoader]:
    dataset = MeteoDataset(
        root_dir=root_dir, location=location, input_len=input_len, output_len=output_len
    )
    split_idx = int(split_ratio * len(dataset))

    train_dataset = Subset(dataset, range(split_idx))
    test_dataset = Subset(dataset, range(split_idx, len(dataset)))

    train_dataloader = DataLoader(
        train_dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers
    )
    test_dataloader = DataLoader(
        test_dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers
    )

    return train_dataloader, test_dataloader


if __name__ == "__main__":
    num_workers = os.cpu_count() or 1
    train_dl, test_dl = create_dataloaders(
        root_dir=Path("data/processed/weather_data"),
        location=["BIALYSTOK", "WARSAW", "WROCLAW", "KRAKOW", "POZNAN"],
        input_len=32,
        output_len=8,
        split_ratio=0.8,
        batch_size=16,
        num_workers=num_workers,
    )
    print(f"Number of batches in train dataloader: {len(train_dl)}")
    print(f"Number of batches in test dataloader: {len(test_dl)}")
    print(f"Number of samples in train dataloader: {len(train_dl.dataset)}")
    print(f"Number of samples in test dataloader: {len(test_dl.dataset)}")
    print(f"Number of samples in the dataset: {len(train_dl.dataset) + len(test_dl.dataset)}")
    sample = next(iter(train_dl))
    print(sample[0].shape, sample[1].shape)
    sample = next(iter(test_dl))
    print(sample[0].shape, sample[1].shape)
