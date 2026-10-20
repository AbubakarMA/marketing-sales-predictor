import streamlit as st
import joblib
import pandas as pd
from pathlib import Path

# Locate the saved model relative to app.py
BASE_DIR = Path(__file__).resolve().parent
MODEL_PATH = BASE_DIR / "models" / "marketing_model.joblib"


# Load and cache the saved model
@st.cache_resource
def load_model():
    model = joblib.load(MODEL_PATH)
    return model


# Configure the Streamlit page
st.set_page_config(
    page_title="Marketing Sales Predictor",
    layout="wide"
)

# Display the title
st.title("Marketing Sales Predictor")

st.write(
    "Predict sales based on your proposed TV advertising budget."
)

# Get advertising budget from the user
tv_budget = st.number_input(
    "TV Advertising Budget ($1,000s)",
    min_value=0.0,
    max_value=300.0,
    value=100.0,
    step=1.0
)

# Load the saved model
model = load_model()

# Prepare the input with the feature name used during training
input_data = pd.DataFrame({
    "TV": [tv_budget]
})

# Generate the prediction
predicted_sales = model.predict(input_data)[0]

# Display the predicted sales
st.subheader("Predicted Sales")
st.write(f"**{predicted_sales:.2f} thousand units**")