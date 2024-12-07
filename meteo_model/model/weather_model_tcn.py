from meteo_model.model.base_model import BaseWeatherModel
from pytorch_tcn import TCN
from torch import nn
import torch


class WeatherModelTCN(BaseWeatherModel):
    def __init__(
        self,
        num_features: int,
        num_locations: int,
        output_len: int,
        num_channels: list[int],
        kernel_size: int,
        dropout: float,
    ):
        """
        Initialize the WeatherModelTCN using pytorch_tcn.
        Args:
            num_features (int): The number of features in the input data.
            num_locations (int): The number of locations in the input data.
            output_len (int): The number of days to predict.
            num_channels (list): The number of channels for each TCN layer.
            kernel_size (int): The kernel size for the TCN.
            dropout (float): Dropout rate for regularization.
        """
        self.num_channels = num_channels
        self.kernel_size = kernel_size
        self.dropout = dropout
        super(WeatherModelTCN, self).__init__(num_features, num_locations, output_len)

        self.submodels = nn.ModuleList(
            [
                TCN(
                    num_inputs=num_features,
                    num_channels=num_channels,
                    kernel_size=kernel_size,
                    dropout=dropout,
                    input_shape="NLC",
                )
                for _ in range(num_locations)
            ]
        )
        self.final_fc = nn.Linear(num_locations, 1)

    def process_locations(self, x: torch.Tensor) -> torch.Tensor:
        """
        Process input data for each location using the corresponding TCN model.
        """
        outputs = []
        for i, tcn in enumerate(self.submodels):
            location_input = x[:, i, :, :]
            output = tcn(location_input)

            outputs.append(output)

        stacked_outputs = torch.stack(outputs, dim=1)
        return stacked_outputs

    def forward(self, X: torch.Tensor) -> torch.Tensor:
        """
        Iteratively predicts output_len steps for the given input data.
        """
        predictions: list[torch.Tensor] = []
        if X.dim() == 3:
            X = X.unsqueeze(0)

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
