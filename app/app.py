"""Interactive demonstration of the saved dengue forecasting model."""
import logging

import streamlit as st

from inference import DISTRICTS, load_example, load_model, predict_next_week

st.set_page_config(page_title="Sri Lanka Dengue Forecasting", page_icon="🦟", layout="wide")
st.title("Weekly Dengue Forecasting in Sri Lanka")
st.write("Explore a one-week-ahead case estimate using dengue history, seasonality, and weekly climate observations.")
st.info("Educational capstone demonstration for Colombo, Gampaha, Jaffna, Kalutara, and Kandy. This is not a live public-health forecast.")

try:
    example = load_example()
    load_model()
except Exception:
    logging.exception("Unable to initialize the forecasting demo")
    st.error("The forecasting files could not be loaded. Check the installation steps and model files in the README.")
    st.stop()

use_example = st.checkbox("Start with the historical Kandy example", value=True)
if use_example:
    st.caption("Historical observation: 13 December 2025. The saved notebook reports a forecast of 77.44 cases for this example. You can edit the inputs below.")
defaults = example if use_example else {}

with st.form("forecast"):
    st.subheader("Forecast settings")
    district_col, week_col = st.columns(2)
    district = district_col.selectbox("District", DISTRICTS, index=DISTRICTS.index(defaults.get("district", "Colombo")))
    week = week_col.number_input("Current epidemiological week", min_value=1, max_value=53, value=int(defaults.get("week", 1)), step=1)
    st.subheader("Dengue case history")
    st.caption("Reported case counts for the current week and the specified earlier weeks.")
    inputs = {"district": district, "week": week}
    for column, (field, label) in zip(st.columns(4), [
        ("cases", "Current week"), ("cases_lag_1", "1 week ago"),
        ("cases_lag_2", "2 weeks ago"), ("cases_lag_4", "4 weeks ago"),
    ]):
        inputs[field] = column.number_input(label, min_value=0, value=int(defaults.get(field, 0)), step=1, key=f"{use_example}_{field}")

    st.subheader("Weekly climate history")
    st.caption("Use weekly rainfall totals, weekly mean temperature, and weekly mean relative humidity, matching the observation weeks above.")
    for prefix, label, initial in [
        ("rainfall_total", "Rainfall total (mm/week)", 0.0),
        ("temperature_mean", "Mean temperature (°C)", 25.0),
        ("humidity_mean", "Mean relative humidity (%)", 75.0),
    ]:
        st.markdown(f"**{label}**")
        for column, (suffix, period) in zip(st.columns(5), [
            ("", "Current week"), ("_lag_1", "1 week ago"),
            ("_lag_2", "2 weeks ago"), ("_lag_4", "4 weeks ago"), ("_lag_8", "8 weeks ago"),
        ]):
            field = prefix + suffix
            bounds = {"min_value": 0.0} if prefix == "rainfall_total" else {}
            if prefix == "humidity_mean":
                bounds = {"min_value": 0.0, "max_value": 100.0}
            inputs[field] = column.number_input(period, value=float(defaults.get(field, initial)), key=f"{use_example}_{field}", **bounds)
    submitted = st.form_submit_button("Predict next week", type="primary", use_container_width=True)

if submitted:
    try:
        prediction = predict_next_week(inputs)
    except ValueError as exc:
        st.error(str(exc))
    except Exception:
        logging.exception("Forecast failed")
        st.error("The forecast could not be generated. Check your inputs and restart the app if the issue persists.")
    else:
        st.success("Forecast generated successfully.")
        st.metric(f"Estimated cases for {district} next week", f"{round(prediction):,} cases")
        st.caption(f"Unrounded model estimate: {prediction:.2f} cases. This is a point estimate; prediction intervals are not provided.")

st.divider()
st.caption("Capstone Project II · Weekly Dengue Forecasting in Sri Lanka")
