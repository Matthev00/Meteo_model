import streamlit as st
import pandas as pd
import plotly.express as px
from meteo_model.data.config import COLUMNS
import requests


API_URL = "http://127.0.0.1:8000/predict"


def predict_via_api(n_days: int) -> pd.DataFrame:
    response = requests.post(API_URL, json={"n_days": n_days})
    if response.status_code != 200:
        st.error(f"API error: {response.json()['detail']}")
        return pd.DataFrame()
    return pd.DataFrame(response.json())


def visualize_predictions(preds: pd.DataFrame, selected_parameters: list[str], days_forward: int):
    param_names = {
        "tavg": ("°C", "Average temperature"),
        "tmin": ("°C", "Minimum temperature"),
        "tmax": ("°C", "Maximum temperature"),
        "prcp": ("mm", "Precipitation"),
        "snow": ("mm", "Snow"),
        "wdir": ("°" ,"Wind direction"),
        "wspd": ("km/h", "Wind speed"),
        "pres": ("hPa", "Pressure"),
    }

    st.subheader("Visualization")

    num_columns = 2
    cols = st.columns(num_columns)

    for i, param in enumerate(selected_parameters):
        col = cols[i % num_columns]
        with col:
            if days_forward == 1:
                fig = px.scatter(preds, x="date", y=param, title=param_names[param][1])
            else:
                fig = px.line(preds, x="date", y=param, title=param_names[param][1])
            fig.update_layout(xaxis_title="Date (DD-MM)", yaxis_title=param_names[param][0])
            st.plotly_chart(fig)


def user_input_features() -> tuple[list[str], int]:
    st.sidebar.title("Vizualization settings")
    parameters = COLUMNS.copy()
    parameters.remove("snow")
    selected_parameters = st.sidebar.multiselect(
        "Select parameters to visualize:", parameters, default=["tavg"]
    )
    if not selected_parameters:
        st.warning("Please select at least one parameter to visualize.")
        return [], 0
    days_forward = st.sidebar.slider("Number of time steps to predict: ", 1, 8, 4)

    return selected_parameters, days_forward


def main():
    st.title("Weather forecast")
    st.write(
        "This is a simple weather forecast app. You can select parameters to visualize and the number of days to predict."
    )

    selected_parameters, days_forward = user_input_features()

    preds = predict_via_api(days_forward)

    visualize_predictions(preds, selected_parameters, days_forward)


if __name__ == "__main__":
    main()
