from abc import ABC, abstractmethod
from torch import nn
import torch


class BaseWeatherModel(ABC, nn.Module):
    def __init__(self, num_features, num_locations: int, output_len: int):
        """
        Args:
            num_features (int): The number of features in the input data.
            num_locations (int): The number of locations in the input data.
            output_len (int): The number of days to predict.
        """
        super(BaseWeatherModel, self).__init__()
        self.num_features = num_features
        self.num_locations = num_locations
        self.output_len = output_len

    @abstractmethod
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        pass
