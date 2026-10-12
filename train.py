from pathlib import Path
import json
import warnings
import joblib
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder
from sklearn.tree import DecisionTreeRegressor

warnings.filterwarnings("ignore", category=FutureWarning)
BASE_DIR = Path(__file__).resolve().parent
DATA_PATH = BASE_DIR / "train.csv"
ARTIFACT_DIR = BASE_DIR / "artifacts"
ARTIFACT_DIR.mkdir(exist_ok=True)
TARGET = "Price_INR_Lakhs"
ID_COLUMN = "Property_ID"
FEATURES = [
    "City", "Locality_Type", "Property_Type", "BHK", "Bathrooms",
    "Super_Area_SqFt", "Carpet_Area_SqFt", "Floor_Number", "Total_Floors",
    "Age_of_Property", "Furnishing_Status", "Parking", "Lift_Available",
    "Gated_Community", "Distance_to_Metro_km", "Distance_to_City_Center_km",
]
CATEGORICAL = ["City", "Locality_Type", "Property_Type", "Furnishing_Status"]
NUMERIC = [column for column in FEATURES if column not in CATEGORICAL]

def calculate_metrics(actual, predicted):
    return {
        "r2": float(r2_score(actual, predicted)),
        "mae_lakhs": float(mean_absolute_error(actual, predicted)),
        "rmse_lakhs": float(np.sqrt(mean_squared_error(actual, predicted))),
    }

def main():
    if not DATA_PATH.exists():
        raise FileNotFoundError(f"Place your dataset at: {DATA_PATH}")
    df = pd.read_csv(DATA_PATH)
    print("Dataset shape:", df.shape)
    print("Total missing values:", int(df.isna().sum().sum()))
    print("Duplicate rows:", int(df.duplicated().sum()))
    missing = [col for col in FEATURES + [TARGET] if col not in df.columns]
    if missing:
        raise ValueError(f"Required columns missing from train.csv: {missing}")
    df = df.drop(columns=[ID_COLUMN], errors="ignore")
    df = df[FEATURES + [TARGET]].copy()
    df[TARGET] = pd.to_numeric(df[TARGET], errors="coerce")
    df = df.dropna(subset=[TARGET])
    df[NUMERIC] = df[NUMERIC].replace([np.inf, -np.inf], np.nan)
    X, y = df[FEATURES], df[TARGET]

    # 70% train, 15% validation, 15% final test.
    X_dev, X_test, y_dev, y_test = train_test_split(X, y, test_size=0.15, random_state=20)
    X_train, X_val, y_train, y_val = train_test_split(
        X_dev, y_dev, test_size=(0.15 / 0.85), random_state=20
    )

    categorical_pipeline = Pipeline(steps=[
        ("imputer", SimpleImputer(strategy="most_frequent")),
        ("onehot", OneHotEncoder(handle_unknown="ignore", sparse_output=False)),
    ])
    numeric_pipeline = Pipeline(steps=[("imputer", SimpleImputer(strategy="median"))])
    preprocessor = ColumnTransformer(transformers=[
        ("categorical", categorical_pipeline, CATEGORICAL),
        ("numeric", numeric_pipeline, NUMERIC),
    ])
    models = {
        "Linear Regression": LinearRegression(),
        "Decision Tree": DecisionTreeRegressor(max_depth=20, min_samples_leaf=4, random_state=20),
        "Random Forest": RandomForestRegressor(
            n_estimators=150, max_depth=20, min_samples_split=10,
            min_samples_leaf=4, random_state=20, n_jobs=-1
        ),
        "Gradient Boosting": GradientBoostingRegressor(
            n_estimators=300, learning_rate=0.05, max_depth=3,
            min_samples_split=10, min_samples_leaf=4, random_state=20
        ),
    }
    validation_results = []
    print("\n--- Validation results (used for model selection) ---")
    for name, model in models.items():
        pipeline = Pipeline(steps=[("preprocessor", preprocessor), ("model", model)])
        pipeline.fit(X_train, y_train)
        predictions = pipeline.predict(X_val)
        metrics = calculate_metrics(y_val, predictions)
        validation_results.append({"model": name, **metrics})
        print(f"{name}: R²={metrics['r2']:.4f}, MAE={metrics['mae_lakhs']:.2f} lakh, RMSE={metrics['rmse_lakhs']:.2f} lakh")

    validation_results.sort(key=lambda item: item["r2"], reverse=True)
    best_name = validation_results[0]["model"]
    print("\nSelected using validation R²:", best_name)

    # Refit selected model on train + validation. Keep test set untouched until now.
    X_final = pd.concat([X_train, X_val], axis=0)
    y_final = pd.concat([y_train, y_val], axis=0)
    final_pipeline = Pipeline(steps=[("preprocessor", preprocessor), ("model", models[best_name])])
    final_pipeline.fit(X_final, y_final)
    test_predictions = final_pipeline.predict(X_test)
    test_metrics = calculate_metrics(y_test, test_predictions)
    print("\n--- Final test results ---")
    print("Selected model:", best_name)
    print(f"Testing R²: {test_metrics['r2']:.4f}")
    print(f"MAE: {test_metrics['mae_lakhs']:.2f} lakh")
    print(f"RMSE: {test_metrics['rmse_lakhs']:.2f} lakh")

    joblib.dump(final_pipeline, ARTIFACT_DIR / "house_price_pipeline.joblib")
    metadata = {
        "target": TARGET, "features": FEATURES, "categorical_features": CATEGORICAL,
        "numeric_features": NUMERIC, "selected_model": best_name, "categorical_options": {}
    }
    for column in CATEGORICAL:
        metadata["categorical_options"][column] = sorted(X[column].dropna().astype(str).unique().tolist())
    (ARTIFACT_DIR / "model_metadata.json").write_text(json.dumps(metadata, indent=2), encoding="utf-8")
    output = {"validation_results": validation_results, "selected_model": best_name,
              "test_metrics": test_metrics, "rows_used": int(len(df))}
    (ARTIFACT_DIR / "model_metrics.json").write_text(json.dumps(output, indent=2), encoding="utf-8")
    print("\nModel and metadata saved in artifacts/.")

if __name__ == "__main__":
    main()
