import streamlit as st
import joblib
import pandas as pd
from pathlib import Path
import os
import requests


# --------------------------------------------------
# 1. Locate the saved model
# --------------------------------------------------

BASE_DIR = Path(__file__).resolve().parent
MODEL_PATH = BASE_DIR / "models" / "marketing_model.joblib"


# --------------------------------------------------
# 2. Load and cache the saved model
# --------------------------------------------------

@st.cache_resource
def load_model():
    model = joblib.load(MODEL_PATH)
    return model


# --------------------------------------------------
# 3. Configure the Streamlit page
# --------------------------------------------------

st.set_page_config(
    page_title="Marketing Sales Predictor",
    layout="wide"
)


# --------------------------------------------------
# 4. Display the application title
# --------------------------------------------------

st.title("Marketing Sales Predictor")

st.write(
    "Predict sales based on your proposed TV advertising budget."
)


# --------------------------------------------------
# 5. Get the TV advertising budget from the user
# --------------------------------------------------

tv_budget = st.number_input(
    "TV Advertising Budget ($1,000s)",
    min_value=0.0,
    max_value=300.0,
    value=100.0,
    step=1.0
)


# --------------------------------------------------
# 6. Generate a prediction using the local model
# --------------------------------------------------

model = load_model()

# Prepare the input using the feature name used during training
input_data = pd.DataFrame({
    "TV": [tv_budget]
})

# Generate the local prediction
predicted_sales = float(model.predict(input_data)[0])


# --------------------------------------------------
# 7. Display the local model prediction
# --------------------------------------------------

st.subheader("Predicted Sales")

st.write(
    f"**{predicted_sales:.2f} thousand units**"
)


# --------------------------------------------------
# 8. Configure the FastAPI URL
# --------------------------------------------------

API_URL = os.getenv(
    "API_URL",
    "http://127.0.0.1:8000"
)


# --------------------------------------------------
# 9. Create a function to call FastAPI
# --------------------------------------------------

def get_prediction(tv_budget: float) -> float:
    """Request a sales prediction from FastAPI."""

    endpoint = f"{API_URL.rstrip('/')}/predict"

    payload = {
        "tv_budget": tv_budget
    }

    # Send a POST request to FastAPI
    response = requests.post(
        endpoint,
        json=payload,
        timeout=60
    )

    # Raise an exception if the API returns an HTTP error
    response.raise_for_status()

    # Convert the response to a Python dictionary
    result = response.json()

    # Return the predicted sales value
    return float(result["predicted_sales"])


# --------------------------------------------------
# 10. Compare the local prediction with the API prediction
# --------------------------------------------------

st.subheader("Prediction Comparison")

try:
    # Get a prediction from FastAPI
    api_prediction = get_prediction(float(tv_budget))

    # Display both predictions side by side
    col1, col2 = st.columns(2)

    with col1:
        st.metric(
            label="Local Model Prediction",
            value=f"{predicted_sales:.2f} thousand units"
        )

    with col2:
        st.metric(
            label="FastAPI Prediction",
            value=f"{api_prediction:.2f} thousand units"
        )

    # Calculate the difference between the predictions
    difference = abs(predicted_sales - api_prediction)

    st.write(
        f"**Absolute difference:** {difference:.4f} thousand units"
    )

    # Indicate whether the predictions match
    if difference < 0.01:
        st.success(
            "Both models produce approximately the same prediction."
        )
    else:
        st.warning(
            "The predictions differ. Check that both environments "
            "use the same trained model."
        )

except requests.RequestException as error:
    st.error(
        f"Could not retrieve a prediction from FastAPI: {error}"
    )

except (KeyError, ValueError) as error:
    st.error(
        f"Could not process the API response: {error}"
    )