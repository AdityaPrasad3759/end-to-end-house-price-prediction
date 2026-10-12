from pathlib import Path
import json
import joblib
import pandas as pd
import streamlit as st

BASE_DIR = Path(__file__).resolve().parent
ARTIFACT_DIR = BASE_DIR / "artifacts"
MODEL_PATH = ARTIFACT_DIR / "house_price_pipeline.joblib"
METADATA_PATH = ARTIFACT_DIR / "model_metadata.json"
st.set_page_config(page_title="House Price Predictor", page_icon="🏠")
st.title("🏠 House Price Prediction")
st.write("Enter property details to estimate its price. This is not a formal valuation.")
if not MODEL_PATH.exists() or not METADATA_PATH.exists():
    st.error("Model files are missing. Add train.csv and run `python train.py` first.")
    st.stop()
model = joblib.load(MODEL_PATH)
metadata = json.loads(METADATA_PATH.read_text(encoding="utf-8"))
options = metadata["categorical_options"]
with st.form("property_form"):
    st.subheader("Location and property type")
    city = st.selectbox("City", options["City"])
    locality = st.selectbox("Locality type", options["Locality_Type"])
    property_type = st.selectbox("Property type", options["Property_Type"])
    furnishing = st.selectbox("Furnishing status", options["Furnishing_Status"])
    st.subheader("Property details")
    c1, c2 = st.columns(2)
    with c1:
        bhk = st.number_input("BHK", min_value=1, max_value=20, value=2)
        bathrooms = st.number_input("Bathrooms", min_value=1, max_value=20, value=2)
        super_area = st.number_input("Super area (sq ft)", min_value=1.0, value=1000.0)
        carpet_area = st.number_input("Carpet area (sq ft)", min_value=1.0, value=800.0)
        floor = st.number_input("Floor number", min_value=0, max_value=200, value=1)
        total_floors = st.number_input("Total floors", min_value=1, max_value=200, value=5)
    with c2:
        age = st.number_input("Age of property (years)", min_value=0, max_value=300, value=5)
        parking = st.number_input("Parking (use dataset's numeric encoding)", min_value=0.0, value=1.0)
        lift = st.selectbox("Lift available (dataset encoding)", [0, 1], index=1)
        gated = st.selectbox("Gated community (dataset encoding)", [0, 1], index=1)
        metro = st.number_input("Distance to metro (km)", min_value=0.0, value=2.0)
        city_center = st.number_input("Distance to city center (km)", min_value=0.0, value=5.0)
    submitted = st.form_submit_button("Estimate price")
if submitted:
    if carpet_area > super_area:
        st.warning("Carpet area is greater than super area. Please verify the values.")
    row = pd.DataFrame([{
        "City": city, "Locality_Type": locality, "Property_Type": property_type,
        "BHK": bhk, "Bathrooms": bathrooms, "Super_Area_SqFt": super_area,
        "Carpet_Area_SqFt": carpet_area, "Floor_Number": floor,
        "Total_Floors": total_floors, "Age_of_Property": age,
        "Furnishing_Status": furnishing, "Parking": parking,
        "Lift_Available": lift, "Gated_Community": gated,
        "Distance_to_Metro_km": metro, "Distance_to_City_Center_km": city_center,
    }])
    prediction = float(model.predict(row)[0])
    st.success(f"Estimated price: ₹{prediction:,.2f} lakh")
    st.caption(f"Approximately ₹{prediction / 100:,.3f} crore. Selected model: {metadata['selected_model']}.")
