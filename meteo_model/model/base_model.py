from abc import ABC, abstractmethod
from torch import nn
import torch


class BaseWeatherModel(ABC, nn.Module):
    def __init__(self, input_size, output_size, num_locations):
        super(BaseWeatherModel, self).__init__()
        self.input_size = input_size
        self.output_size = output_size
        self.num_locations = num_locations

    @abstractmethod
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        pass
