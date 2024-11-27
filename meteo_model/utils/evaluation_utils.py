import matplotlib.pyplot as plt
import numpy as np


def visualize_predictions(X, y, pred, features_names: list[str]):

    input_len = X.shape[1]
    output_len = y.shape[1]
    num_features = X.shape[2]

    time_input = np.arange(input_len)
    time_output = np.arange(input_len, input_len + output_len)

    fig, axs = plt.subplots(num_features, 1, figsize=(10, 5 * num_features))

    for feature_idx in range(num_features):
        ax = axs[feature_idx] if num_features > 1 else axs

        ax.plot(time_input, X[0, :, feature_idx], label="Input", marker="o")

        ax.plot(time_output, y[0, :, feature_idx], label="True Output", marker="x")

        ax.plot(time_output, pred[0, :, feature_idx], label="Predicted Output", marker="s")

        ax.set_title(f"Feature {features_names[feature_idx]}")
        ax.set_xlabel("Time Steps")
        ax.legend()
        ax.grid()

    plt.tight_layout()
    plt.show()
