# app.py
# House Price Predictor - Streamlit Web App

import streamlit as st
import pickle
import numpy as np

st.set_page_config(
    page_title="House Price Predictor",
    page_icon=":house:",
    layout="centered"
)

st.title("House Price Predictor")
st.write("Adjust the parameters below to estimate property value")
st.divider()

@st.cache_resource
def load_model():
    with open('model.pkl', 'rb') as f:
        model = pickle.load(f)
    return model

model = load_model()

st.subheader("Property Details")

col1, col2 = st.columns(2)

with col1:
    area = st.slider(
        "Area (sq ft)",
        min_value=1000,
        max_value=20000,
        value=5000,
        step=100
    )
    bedrooms = st.slider(
        "Bedrooms",
        min_value=1,
        max_value=6,
        value=3,
        step=1
    )
    bathrooms = st.slider(
        "Bathrooms",
        min_value=1,
        max_value=4,
        value=2,
        step=1
    )

with col2:
    stories = st.slider(
        "Stories",
        min_value=1,
        max_value=4,
        value=2,
        step=1
    )
    parking = st.slider(
        "Parking Spaces",
        min_value=0,
        max_value=3,
        value=1,
        step=1
    )
    prefarea = st.selectbox(
        "Preferred Area",
        options=["No", "Yes"]
    )
    airconditioning = st.selectbox(
        "Air Conditioning",
        options=["No", "Yes"]
    )

st.divider()

if st.button("Predict Price", type="primary"):
    prefarea_val = 1 if prefarea == "Yes" else 0
    ac_val = 1 if airconditioning == "Yes" else 0
    
    # Default values for features not in UI
    mainroad_yes = 1
    guestroom_yes = 0
    basement_yes = 0
    hotwaterheating_yes = 0
    furnishingstatus_semi = 0
    furnishingstatus_unfurnished = 0
    
    # Engineered features
    area_per_room = area / (bedrooms + 1)
    price_per_sqft = 0  # Not used for prediction, but model expects it
    
    # 15 features matching training data
    features = np.array([[
        area,
        bedrooms,
        bathrooms,
        stories,
        parking,
        mainroad_yes,
        guestroom_yes,
        basement_yes,
        hotwaterheating_yes,
        ac_val,
        prefarea_val,
        furnishingstatus_semi,
        furnishingstatus_unfurnished,
        area_per_room,
        price_per_sqft
    ]])
    
    prediction_usd = model.predict(features)[0]
    prediction_inr = prediction_usd * 83
    
    st.success("### Estimated Price")
    
    col1, col2 = st.columns(2)
    with col1:
        st.metric(
            label="USD",
            value=f"${prediction_usd:,.2f}"
        )
    with col2:
        st.metric(
            label="INR (approx)",
            value=f"Rs. {prediction_inr:,.2f}"
        )

st.divider()
st.caption("Model: Random Forest | R2 Score: 93.39%")