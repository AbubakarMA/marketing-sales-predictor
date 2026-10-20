import logging
import os

import requests
import streamlit as st

logger = logging.getLogger(__name__)

st.set_page_config(page_title="Marketing Sales Predictor", layout="centered")
st.title("Marketing Sales Predictor")
st.write("Estimate sales from a TV advertising budget using a deployed ML inference API.")
st.caption("Educational demonstration, not a validated commercial sales forecast.")

API_URL = os.environ.get("API_URL", "http://127.0.0.1:8000").rstrip("/")


def get_prediction(tv_budget: float) -> float:
    """Request a sales prediction from the FastAPI service."""
    response = requests.post(
        f"{API_URL}/predict",
        json={"tv_budget": float(tv_budget)},
        timeout=60,
    )
    response.raise_for_status()
    result = response.json()
    return float(result["predicted_sales"])


with st.form("prediction_form"):
    tv_budget = st.number_input(
        "TV Advertising Budget ($1,000s)",
        min_value=0.0,
        max_value=300.0,
        value=100.0,
        step=1.0,
    )
    submitted = st.form_submit_button("Predict Sales")

if submitted:
    with st.spinner("Requesting prediction..."):
        try:
            prediction = get_prediction(tv_budget)
        except requests.RequestException:
            logger.exception("Prediction API request failed")
            st.error("The prediction service is temporarily unavailable. Please try again.")
        except (KeyError, TypeError, ValueError):
            logger.exception("Invalid prediction API response")
            st.error("The prediction service returned an unexpected response. Please try again.")
        else:
            st.subheader("Predicted Sales")
            st.metric("Estimated sales", f"{prediction:.2f} thousand units")
