from flask import Flask, request, jsonify
import torch
from utils import prepare_pred_df, get_dates
from meteo_model.utils.model_utils import load_model
from meteo_model.data.api.api_data_provider import get_weather_tensor_for_days

app = Flask(__name__)


@app.route("/")
def read_root():
    return {"message": "Welcome to the Meteo Forecasting API"}


@app.route("/predict", methods=["POST"])
def predict():
    data = request.get_json()
    n_days = data.get("n_days")

    if n_days not in range(1, 9):
        return jsonify({"detail": "Number of days must be between 1 and 8"}), 400

    try:
        device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

        model_for_days = {
            1: ("MeteoModel-1_day", 2),
            2: ("MeteoModel-2_days", 1),
            3: ("MeteoModel-3_days", 1),
            4: ("MeteoModel-4_days", 1),
            5: ("MeteoModel-5_days", 1),
            6: ("MeteoModel-6_days", 2),
            7: ("MeteoModel-7_days", 2),
            8: ("MeteoModel-8_days", 2),
        }
        input_len = {
            1: 16,
            2: 6,
            3: 5,
            4: 10,
            5: 28,
            6: 6,
            7: 12,
            8: 7,
        }
        X, pred_end_day = get_weather_tensor_for_days(
            input_len[n_days], ["WARSAW", "WROCLAW", "POZNAN", "KRAKOW", "BIALYSTOK"]
        )

        model = load_model(*model_for_days[n_days], map_location=device)
        model.eval()

        with torch.inference_mode():
            pred = model(X.to(device))[0].detach().cpu().numpy()

        preds = prepare_pred_df(pred).iloc[:n_days, :]
        preds["date"] = get_dates(pred_end_day, n_days)

        return jsonify(preds.to_dict(orient="records"))

    except Exception as e:
        return jsonify({"detail": f"Prediction failed: {str(e)}"}), 500


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=8000, debug=True)
