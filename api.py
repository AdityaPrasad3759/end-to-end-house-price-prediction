from pathlib import Path
import joblib
import pandas as pd
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

BASE_DIR = Path(__file__).resolve().parent
MODEL_PATH = BASE_DIR / "artifacts" / "house_price_pipeline.joblib"
app = FastAPI(title="House Price Prediction API", version="1.0.0",
              description="Predict estimated property price in INR lakhs.")

class PropertyInput(BaseModel):
    City: str
    Locality_Type: str
    Property_Type: str
    BHK: float = Field(gt=0)
    Bathrooms: float = Field(gt=0)
    Super_Area_SqFt: float = Field(gt=0)
    Carpet_Area_SqFt: float = Field(gt=0)
    Floor_Number: float
    Total_Floors: float = Field(gt=0)
    Age_of_Property: float = Field(ge=0)
    Furnishing_Status: str
    Parking: float = Field(ge=0)
    Lift_Available: float = Field(ge=0)
    Gated_Community: float = Field(ge=0)
    Distance_to_Metro_km: float = Field(ge=0)
    Distance_to_City_Center_km: float = Field(ge=0)

@app.get("/")
def root():
    return {"message": "House Price Prediction API is running", "docs": "/docs",
            "model_ready": MODEL_PATH.exists()}

@app.get("/health")
def health():
    return {"status": "ok", "model_ready": MODEL_PATH.exists()}

@app.post("/predict")
def predict(property_data: PropertyInput):
    if not MODEL_PATH.exists():
        raise HTTPException(status_code=503, detail="Model missing. Run python train.py first.")
    model = joblib.load(MODEL_PATH)
    row = pd.DataFrame([property_data.model_dump()])
    price = float(model.predict(row)[0])
    return {"predicted_price_inr_lakhs": round(price, 2),
            "predicted_price_inr_crore": round(price / 100, 3),
            "note": "Estimate only; not a formal property valuation."}
