# House Price Prediction — End-to-End ML Project

A beginner-friendly project that trains and evaluates regression models, saves the complete preprocessing/model pipeline, serves predictions through FastAPI, and provides a Streamlit UI.

## Dataset
Put your `train.csv` in this folder. Required target: `Price_INR_Lakhs`.

Expected features: `City`, `Locality_Type`, `Property_Type`, `BHK`, `Bathrooms`, `Super_Area_SqFt`, `Carpet_Area_SqFt`, `Floor_Number`, `Total_Floors`, `Age_of_Property`, `Furnishing_Status`, `Parking`, `Lift_Available`, `Gated_Community`, `Distance_to_Metro_km`, `Distance_to_City_Center_km`. `Property_ID` is excluded as an identifier.

## Setup
Python 3.10+ recommended.

```bash
python -m venv .venv
```
Windows: `.venv\Scripts\activate`  
macOS/Linux: `source .venv/bin/activate`

```bash
pip install -r requirements.txt
python train.py
```

The training script creates train/validation/test splits, fits preprocessing on training data only, compares Linear Regression, Decision Tree, Random Forest and Gradient Boosting using validation R², refits the selected model on train+validation, and evaluates once on a held-out test set. It saves the fitted pipeline and metadata under `artifacts/`.

## Streamlit UI
After training:
```bash
streamlit run app.py
```

## FastAPI
```bash
uvicorn api:app --reload
```
Open `http://127.0.0.1:8000/docs` and test `POST /predict`.

## Docker
After training and creating artifacts:
```bash
docker build -t house-price-predictor .
docker run -p 8000:8000 house-price-predictor
```

## Important
- R² is not an accuracy percentage. R² 0.78 means roughly 78% of target variance is explained relative to a mean-prediction baseline on that test set.
- MAE and RMSE are in lakhs because the target is `Price_INR_Lakhs`.
- 90%+ R² cannot be promised; it depends on data quality and available features.
- Do not remove expensive properties just because errors are large; investigate first.
- Model comparison uses validation data. The held-out test set is used for final evaluation.
- Results can differ from the notebook because this project uses a separate 70/15/15 train/validation/test split.
