import streamlit as st
import pandas as pd
import plotly.express as px
from meteo_model.data.config import COLUMNS
from meteo_model.utils.app_utils import predict, get_dates


def visualize_predictions(preds: pd.DataFrame, selected_parameters: list[str], days_forward: int):
    param_names = {
        "tavg": "Average temperature",
        "tmin": "Minimum temperature",
        "tmax": "Maximum temperature",
        "prcp": "Precipitation",
        "snow": "Snow",
        "wdir": "Wind direction",
        "wspd": "Wind speed",
        "pres": "Pressure",
    }

    st.subheader("Visualization")

    num_columns = 2
    cols = st.columns(num_columns)

    for i, param in enumerate(selected_parameters):
        col = cols[i % num_columns]
        with col:
            if days_forward == 1:
                fig = px.scatter(preds, x="date", y=param, title=param_names[param])
            else:
                fig = px.line(preds, x="date", y=param, title=param_names[param])
            fig.update_layout(xaxis_title="Date (DD-MM)")
            st.plotly_chart(fig)


def user_input_features() -> tuple[list[str], int]:
    st.sidebar.title("Vizualization settings")
    parameters = COLUMNS
    selected_parameters = st.sidebar.multiselect(
        "Select parameters to visualize:", parameters, default=["tavg"]
    )
    if not selected_parameters:
        st.warning("Please select at least one parameter to visualize.")
        return [], 0
    days_forward = st.sidebar.slider("Number of time steps to predict: ", 1, 4, 4)

    return selected_parameters, days_forward


def main():
    st.title("Weather forecast")
    st.write(
        "This is a simple weather forecast app. You can select parameters to visualize and the number of days to predict."
    )

    selected_parameters, days_forward = user_input_features()

    preds = predict(days_forward)
    preds["date"] = get_dates(days_forward)

    visualize_predictions(preds, selected_parameters, days_forward)


if __name__ == "__main__":
    main()
