# app.py - Updated to use joblib
import streamlit as st
import joblib
import numpy as np

st.set_page_config(
    page_title="House Price Predictor",
    page_icon=":house:",
    layout="centered"
)

st.title("House Price Predictor")
st.write("Adjust the parameters below to estimate property value")
st.divider()

# Load model with joblib
@st.cache_resource
def load_model():
    return joblib.load('model.joblib')

model = load_model()

st.subheader("Property Details")

col1, col2 = st.columns(2)

with col1:
    area = st.slider("Area (sq ft)", 1000, 20000, 5000, 100)
    bedrooms = st.slider("Bedrooms", 1, 6, 3, 1)
    bathrooms = st.slider("Bathrooms", 1, 4, 2, 1)

with col2:
    stories = st.slider("Stories", 1, 4, 2, 1)
    parking = st.slider("Parking Spaces", 0, 3, 1, 1)
    prefarea = st.selectbox("Preferred Area", ["No", "Yes"])
    airconditioning = st.selectbox("Air Conditioning", ["No", "Yes"])

st.divider()

if st.button("Predict Price", type="primary"):
    prefarea_val = 1 if prefarea == "Yes" else 0
    ac_val = 1 if airconditioning == "Yes" else 0
    
    mainroad_yes = 1
    guestroom_yes = 0
    basement_yes = 0
    hotwaterheating_yes = 0
    furnishingstatus_semi = 0
    furnishingstatus_unfurnished = 0
    
    area_per_room = area / (bedrooms + 1)
    price_per_sqft = 0
    
    features = np.array([[
        area, bedrooms, bathrooms, stories, parking,
        mainroad_yes, guestroom_yes, basement_yes,
        hotwaterheating_yes, ac_val, prefarea_val,
        furnishingstatus_semi, furnishingstatus_unfurnished,
        area_per_room, price_per_sqft
    ]])
    
    prediction_usd = model.predict(features)[0]
    prediction_inr = prediction_usd * 83
    
    st.success("### Estimated Price")
    
    col1, col2 = st.columns(2)
    with col1:
        st.metric("USD", f"${prediction_usd:,.2f}")
    with col2:
        st.metric("INR (approx)", f"Rs. {prediction_inr:,.2f}")

st.divider()
st.caption("Model: Random Forest | R2 Score: 93.39%")