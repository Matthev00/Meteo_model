from meteo_model.model.base_model import BaseWeatherModel
from torch.nn import LSTM
from torch import nn
import torch


class WeatherModelLSTM(BaseWeatherModel):
    def __init__(
        self,
        num_features: int,
        num_locations: int,
        output_len: int,
        hidden_size: int,
        num_layers: int,
    ):
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
        self.submodels = nn.ModuleList(
            [
                nn.Sequential(
                    LSTM(num_features, hidden_size, num_layers, batch_first=True),
                    nn.Linear(hidden_size, num_features),
                )
                for _ in range(num_locations)
            ]
        )
        self.final_fc = nn.Linear(num_locations, 1)

    def process_locations(self, x: torch.Tensor) -> torch.Tensor:
        """
        Process input data for each location using the corresponding LSTM model.
        """
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
        """
        Iteratively predicts output_len steps for the given input data.
        """
        predictions: list[torch.Tensor] = []

        for _ in range(self.output_len):
            yhat = self.process_locations(X)
            yhat = yhat[:, :, -1, :]
            predictions.append(yhat.unsqueeze(2))

            new_X = torch.roll(X, shifts=-1, dims=2).clone()
            new_X[:, :, -1, :] = yhat
            X = new_X

        predictions_tensor = torch.cat(predictions, dim=2)

        predictions_tensor = predictions_tensor.permute(0, 2, 3, 1)
        final_output = self.final_fc(predictions_tensor)
        final_output = final_output.permute(0, 3, 1, 2)

        return final_output
