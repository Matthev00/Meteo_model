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
        X = get_weather_tensor_for_days(n_days, ["WARSAW"])  # Change location
        device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

        model_for_days = {
            1: ("Mete-test", 4),
            2: ("Mete-test", 4),
            3: ("Mete-test", 4),
            4: ("Mete-test", 4),
            # 5: ("Mete-test", 4),
            # 6: ("Mete-test", 4),
            # 7: ("Mete-test", 4),
            # 8: ("Mete-test", 4),
        }  # Change models

        model = load_model(*model_for_days[n_days]).to(device)
        model.eval()

        with torch.inference_mode():
            pred = model(X.to(device))[0].detach().cpu().numpy()

        preds = prepare_pred_df(pred).iloc[:n_days, :]
        preds["date"] = get_dates(n_days)

        return jsonify(preds.to_dict(orient="records"))

    except Exception as e:
        return jsonify({"detail": f"Prediction failed: {str(e)}"}), 500

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=8000, debug=True)
