import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


def visualize_predictions(
    X: pd.DataFrame, y: pd.DataFrame, pred: pd.DataFrame, features_names: list[str]
) -> None:

    input_len = X.shape[0]
    output_len = y.shape[0]
    num_features = len(features_names)

    time_input = np.arange(input_len)
    time_output = np.arange(input_len, input_len + output_len)

    fig, axs = plt.subplots(num_features, 1, figsize=(10, 5 * num_features))

    for idx, feature in enumerate(features_names):
        ax = axs[idx] if num_features > 1 else axs

        ax.plot(time_input, X[feature], label="Input", marker="o")

        ax.plot(time_output, y[feature], label="True Output", marker="x")

        ax.plot(time_output, pred[feature], label="Predicted Output", marker="s")

        ax.set_title(feature)
        ax.set_xlabel("Time Steps")
        ax.legend()
        ax.grid()

    plt.tight_layout()
    plt.show()
