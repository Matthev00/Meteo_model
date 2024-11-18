from meteo_model.model.base_model import BaseWeatherModel
from torch.nn import LSTM
from torch import nn
import torch


class WeatherModelLSTM(BaseWeatherModel):
    def __init__(self, num_features: int, num_locations: int, output_len: int, hidden_size: int, num_layers: int):
        """
        Initialize the WeatherModelLSTM.
        Args:
            num_features (int): The number of features in the input data.
            num_locations (int): The number of locations in the input data.
            output_len (int): The number of days to predict.
            hidden_size (int): The number of features in the hidden state of the LSTM.
            num_layers (int): The number of layers in the LSTM.
        """
        super(WeatherModelLSTM, self).__init__(num_features, num_locations, output_len)
        self.submodels = nn.ModuleList([
            nn.Sequential(
                LSTM(num_features, hidden_size, num_layers, batch_first=True),
                nn.Linear(hidden_size, num_features)
            ) for _ in range(num_locations)
        ])

    def forecast(self, x: torch.Tensor) -> torch.Tensor:
        outputs = []
        for i, submodel in enumerate(self.submodels):
            location_input = x[:, i, :, :]  
            lstm, fc = submodel[0], submodel[1]
            output, _ = lstm(location_input)  # (B, L, H)
            final_output = fc(output)  # (B, L, F)
            outputs.append(final_output)
        
        stacked_outputs = torch.stack(outputs, dim=1) 
        return stacked_outputs 

    def forward(self, X: torch.Tensor) -> torch.Tensor:
        batch_size, num_locations, sequence_length, num_features = X.size()

        for _ in range(self.output_len):
            X = X.view(batch_size, num_locations, sequence_length, num_features)
            yhat = self.forecast(X)
            yhat = yhat[:, :, -1, :] 

            new_X = torch.roll(X, shifts=-1, dims=2).clone()
            new_X[:, :, -1, :] = yhat
            X = new_X

        return X