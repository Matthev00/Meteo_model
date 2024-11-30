import torch
from torch.utils.data import Dataset
import pandas as pd
from pathlib import Path
from collections import defaultdict
from typing import Dict, Optional

from meteo_model.data.config import LOCATIONS_NAMES, DATASET_START_YEAR, DATASET_END_YEAR


class MeteoDataset(Dataset):
    def __init__(
        self,
        root_dir: Path = Path("data/processed/weather_data"),
        location: Optional[list[str]] = None,
        target_location: str = LOCATIONS_NAMES[0],
        input_len: int = 32,
        output_len: int = 8,
    ):
        """
        Dataset for weather data.

        Args:
            root_dir (Path): Path to the directory containing the data.
            location (list): List of locations. If empty, all locations will be used.
            target_location (str): Target Location.
            input_len (int): Number of days to look back.
            output_len (int): Number of days to predict.
        """
        if output_len < 1:
            raise ValueError("output_len should be greater than 0.")
        if not root_dir.exists():
            raise ValueError("root_dir does not exist.")
        if root_dir.exists() and not root_dir.is_dir():
            raise ValueError("root_dir should be a directory.")

        self.root_dir = root_dir
        self.input_len = input_len
        self.output_len = output_len

        if location is None:
            location = LOCATIONS_NAMES
        self.location = location
        self.target_location = target_location

        self.start_year = DATASET_START_YEAR
        self.end_year = DATASET_END_YEAR
        self.data = self._load_data()

    def _load_data(self) -> Dict[int, Dict[str, pd.DataFrame]]:
        """
        Load data from the specified directory.
        """
        all_data: Dict[int, Dict[str, pd.DataFrame]] = defaultdict(dict)

        for year in range(self.start_year, self.end_year + 1):
            for loc in self.location:
                file_path = self.root_dir / str(year) / f"{loc}_weather_data.csv"
                if file_path.exists():
                    df = pd.read_csv(file_path)
                    all_data[year][loc] = df

        return all_data

    def __len__(self) -> int:
        total_days = 0
        for year in range(self.start_year, self.end_year + 1):
            total_days += self.data[year][self.location[0]].shape[0]
        return total_days - self.input_len - self.output_len

    def _get_day(self, idx: int) -> tuple[int, int]:
        """
        Get the year and day corresponding to the index.
        """
        day = idx + self.input_len
        year = self.start_year
        while day >= self.data[year][self.location[0]].shape[0]:
            day -= self.data[year][self.location[0]].shape[0]
            year += 1

        return year, day

    def _get_sequence(self, year: int, start_day: int, end_day: int, locations: list[str]) -> list[list[list[float]]]:
        """
        Get the sequence of data for the given year and days.
        Checks if the start_day and end_day are within the bounds of the given year.
        """
        sequence = []
        for loc in locations:
            _end_day = end_day
            if start_day < 0:
                sequence.append(self.data[year - 1][loc].iloc[start_day:, :].values.tolist())
                sequence[-1] += self.data[year][loc].iloc[:_end_day, :].values.tolist()

            elif _end_day > self.data[year][loc].shape[0]:
                sequence.append(self.data[year][loc].iloc[start_day:, :].values.tolist())
                _end_day -= self.data[year][loc].shape[0]
                sequence[-1] += self.data[year + 1][loc].iloc[:_end_day, :].values.tolist()

            else:
                sequence.append(self.data[year][loc].iloc[start_day:_end_day, :].values.tolist())

        return sequence

    def _get_target_sequence(self, year: int, day: int) -> list[list[list[float]]]:
        start_day = day 
        end_day = day + self.output_len
        return self._get_sequence(year, start_day, end_day, [self.target_location])

    def _get_input_sequence(self, year: int, day: int) -> list[list[list[float]]]:
        start_day = day - self.input_len
        end_day = day
        return self._get_sequence(year, start_day, end_day, self.location)

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Get the input and target sequences for the given index.
        Returns:
            tuple[torch.Tensor, torch.Tensor]: input_sequence, target_sequence
            In sizes (num_locations, input_length, num_features)
        """
        year, day = self._get_day(idx)

        input_sequence = torch.tensor(self._get_input_sequence(year, day), dtype=torch.float32)
        target_sequence = torch.tensor(self._get_target_sequence(year, day), dtype=torch.float32)

        return input_sequence, target_sequence


if __name__ == "__main__":
    data = MeteoDataset()
    print(len(data))
    data[327]
