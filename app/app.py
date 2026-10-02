
import os
import math
import joblib
import pandas as pd
import streamlit as st


# ============================================================
# PAGE CONFIGURATION
# ============================================================

st.set_page_config(
    page_title="Sri Lanka Dengue Forecasting",
    page_icon="🦟",
    layout="wide"
)


# ============================================================
# PATHS
# ============================================================

APP_DIR = os.path.dirname(os.path.abspath(__file__))

PROJECT_DIR = os.path.dirname(APP_DIR)

MODEL_DIR = os.path.join(
    PROJECT_DIR,
    "models"
)

MODEL_PATH = os.path.join(
    MODEL_DIR,
    "xgb_climate_model.pkl"
)

PREPROCESSOR_PATH = os.path.join(
    MODEL_DIR,
    "climate_preprocessor.pkl"
)

FEATURES_PATH = os.path.join(
    MODEL_DIR,
    "model_b_features.pkl"
)


# ============================================================
# LOAD MODEL COMPONENTS
# ============================================================

@st.cache_resource
def load_model_components():

    model = joblib.load(MODEL_PATH)

    preprocessor = joblib.load(
        PREPROCESSOR_PATH
    )

    feature_list = joblib.load(
        FEATURES_PATH
    )

    return model, preprocessor, feature_list


try:

    model, preprocessor, feature_list = (
        load_model_components()
    )

except Exception as e:

    st.error(
        "Unable to load the forecasting model."
    )

    st.exception(e)

    st.stop()


# ============================================================
# HEADER
# ============================================================

st.title(
    "🦟 Weekly Dengue Forecasting in Sri Lanka"
)

st.write(
    """
    This application estimates dengue cases for the
    following week using historical dengue surveillance
    information, seasonal patterns, and climate variables.
    """
)

st.info(
    "The forecasting model was developed for Colombo, "
    "Gampaha, Jaffna, Kalutara and Kandy districts."
)


# ============================================================
# FORECAST SETTINGS
# ============================================================

st.header("1. Forecast Settings")

col1, col2 = st.columns(2)

with col1:

    district = st.selectbox(
        "District",
        [
            "Colombo",
            "Gampaha",
            "Jaffna",
            "Kalutara",
            "Kandy"
        ]
    )

with col2:

    week = st.number_input(
        "Current epidemiological week",
        min_value=1,
        max_value=53,
        value=1,
        step=1
    )


# ============================================================
# DENGUE HISTORY
# ============================================================

st.header("2. Dengue Case History")

st.caption(
    "Enter the number of reported dengue cases for "
    "the current and previous required weeks."
)

c1, c2, c3, c4 = st.columns(4)

with c1:

    cases = st.number_input(
        "Current week",
        min_value=0,
        value=0,
        step=1
    )

with c2:

    cases_lag_1 = st.number_input(
        "1 week ago",
        min_value=0,
        value=0,
        step=1
    )

with c3:

    cases_lag_2 = st.number_input(
        "2 weeks ago",
        min_value=0,
        value=0,
        step=1
    )

with c4:

    cases_lag_4 = st.number_input(
        "4 weeks ago",
        min_value=0,
        value=0,
        step=1
    )


# ============================================================
# CLIMATE INPUT FUNCTION
# ============================================================

def climate_inputs(
    title,
    prefix
):

    st.subheader(title)

    a, b, c, d, e = st.columns(5)

    with a:

        current = st.number_input(
            "Current week",
            value=0.0,
            key=f"{prefix}_current"
        )

    with b:

        lag1 = st.number_input(
            "1 week ago",
            value=0.0,
            key=f"{prefix}_lag1"
        )

    with c:

        lag2 = st.number_input(
            "2 weeks ago",
            value=0.0,
            key=f"{prefix}_lag2"
        )

    with d:

        lag4 = st.number_input(
            "4 weeks ago",
            value=0.0,
            key=f"{prefix}_lag4"
        )

    with e:

        lag8 = st.number_input(
            "8 weeks ago",
            value=0.0,
            key=f"{prefix}_lag8"
        )

    return (
        current,
        lag1,
        lag2,
        lag4,
        lag8
    )


# ============================================================
# CLIMATE HISTORY
# ============================================================

st.header("3. Climate History")

st.caption(
    "Provide weekly climate observations corresponding "
    "to the required historical periods."
)

rain = climate_inputs(
    "🌧️ Rainfall",
    "rain"
)

temperature = climate_inputs(
    "🌡️ Mean Temperature",
    "temp"
)

humidity = climate_inputs(
    "💧 Relative Humidity",
    "humidity"
)


# ============================================================
# PREDICTION
# ============================================================

st.divider()

if st.button(
    "🔮 Predict Next Week",
    type="primary",
    use_container_width=True
):

    # Seasonal encoding used during model training
    week_sin = math.sin(
        2 * math.pi * week / 52
    )

    week_cos = math.cos(
        2 * math.pi * week / 52
    )

    input_data = {
        "district": district,

        "cases": cases,
        "cases_lag_1": cases_lag_1,
        "cases_lag_2": cases_lag_2,
        "cases_lag_4": cases_lag_4,

        "week_sin": week_sin,
        "week_cos": week_cos,

        "rainfall_total": rain[0],
        "temperature_mean": temperature[0],
        "humidity_mean": humidity[0],

        "rainfall_total_lag_1": rain[1],
        "rainfall_total_lag_2": rain[2],
        "rainfall_total_lag_4": rain[3],
        "rainfall_total_lag_8": rain[4],

        "temperature_mean_lag_1": temperature[1],
        "temperature_mean_lag_2": temperature[2],
        "temperature_mean_lag_4": temperature[3],
        "temperature_mean_lag_8": temperature[4],

        "humidity_mean_lag_1": humidity[1],
        "humidity_mean_lag_2": humidity[2],
        "humidity_mean_lag_4": humidity[3],
        "humidity_mean_lag_8": humidity[4]
    }

    # Ensure exact training feature order
    input_df = pd.DataFrame(
        [input_data]
    )[feature_list]

    

    try:

        processed_input = (
            preprocessor.transform(
                input_df
            )
        )

        prediction = model.predict(
            processed_input
        )[0]

        prediction = max(
            0.0,
            float(prediction)
        )

        rounded_prediction = int(
            round(prediction)
        )

        st.success(
            "Forecast generated successfully."
        )

        st.metric(
            label=(
                f"Predicted dengue cases for "
                f"{district} next week"
            ),
            value=f"{rounded_prediction} cases"
        )

        st.caption(
            f"Raw model estimate: "
            f"{prediction:.2f} cases"
        )

    except Exception as e:

        st.error(
            "An error occurred while generating "
            "the forecast."
        )

        st.exception(e)


# ============================================================
# FOOTER
# ============================================================

st.divider()

st.caption(
    "Capstone Project II – Weekly Dengue Forecasting "
    "in Sri Lanka"
)
