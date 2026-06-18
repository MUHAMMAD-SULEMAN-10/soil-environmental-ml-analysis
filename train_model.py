"""
train_model.py
Trains two models on soil_data.csv:
1. Clay_pct regressor   -> RandomForestRegressor
2. Fertility_Class classifier -> RandomForestClassifier

Saves both models (with their preprocessing) plus metrics.json
so the Streamlit app can load everything without retraining.
"""

import json
import joblib
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    mean_absolute_error,
    r2_score,
)
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder

df = pd.read_csv("soil_data.csv")

CAT_COLS = ["Land_Use", "Parent_Material"]
NUM_COLS_FOR_CLAY = ["Depth_cm", "Rainfall_mm", "Temperature_C", "Slope_pct"]
NUM_COLS_FOR_FERTILITY = [
    "Organic_Matter_pct", "pH", "Nitrogen_mgkg", "Phosphorus_mgkg",
    "Potassium_mgkg", "CEC_cmolkg", "Clay_pct", "Moisture_pct",
]

# ---------------------------------------------------------------
# MODEL 1: Clay content regression (depth/rainfall/slope -> clay%)
# ---------------------------------------------------------------
X_clay = df[NUM_COLS_FOR_CLAY + CAT_COLS]
y_clay = df["Clay_pct"]

preprocess_clay = ColumnTransformer([
    ("cat", OneHotEncoder(handle_unknown="ignore"), CAT_COLS),
], remainder="passthrough")

clay_pipeline = Pipeline([
    ("prep", preprocess_clay),
    ("model", RandomForestRegressor(n_estimators=200, max_depth=10, random_state=42)),
])

Xc_train, Xc_test, yc_train, yc_test = train_test_split(
    X_clay, y_clay, test_size=0.2, random_state=42
)
clay_pipeline.fit(Xc_train, yc_train)
clay_pred = clay_pipeline.predict(Xc_test)
clay_r2 = r2_score(yc_test, clay_pred)
clay_mae = mean_absolute_error(yc_test, clay_pred)

# ---------------------------------------------------------------
# MODEL 2: Fertility classification (chemistry -> Low/Medium/High)
# ---------------------------------------------------------------
X_fert = df[NUM_COLS_FOR_FERTILITY]
y_fert = df["Fertility_Class"]

fert_pipeline = Pipeline([
    ("model", RandomForestClassifier(n_estimators=200, max_depth=10, random_state=42)),
])

Xf_train, Xf_test, yf_train, yf_test = train_test_split(
    X_fert, y_fert, test_size=0.2, random_state=42, stratify=y_fert
)
fert_pipeline.fit(Xf_train, yf_train)
fert_pred = fert_pipeline.predict(Xf_test)
fert_acc = accuracy_score(yf_test, fert_pred)
fert_report = classification_report(yf_test, fert_pred, output_dict=True)

# Feature importances for fertility model (simple, no one-hot needed)
fert_importances = dict(zip(NUM_COLS_FOR_FERTILITY, fert_pipeline.named_steps["model"].feature_importances_))
fert_importances = dict(sorted(fert_importances.items(), key=lambda x: -x[1]))

# Feature importances for clay model (need expanded feature names from one-hot)
ohe = clay_pipeline.named_steps["prep"].named_transformers_["cat"]
cat_feature_names = list(ohe.get_feature_names_out(CAT_COLS))
all_clay_feature_names = cat_feature_names + NUM_COLS_FOR_CLAY
clay_importances = dict(zip(all_clay_feature_names, clay_pipeline.named_steps["model"].feature_importances_))
clay_importances = dict(sorted(clay_importances.items(), key=lambda x: -x[1]))

# ---------------------------------------------------------------
# Save everything
# ---------------------------------------------------------------
joblib.dump(clay_pipeline, "clay_model.pkl")
joblib.dump(fert_pipeline, "fertility_model.pkl")

metrics = {
    "clay_model": {
        "r2": round(clay_r2, 4),
        "mae": round(clay_mae, 3),
        "features": NUM_COLS_FOR_CLAY + CAT_COLS,
        "feature_importance": {k: round(v, 4) for k, v in clay_importances.items()},
    },
    "fertility_model": {
        "accuracy": round(fert_acc, 4),
        "classification_report": fert_report,
        "features": NUM_COLS_FOR_FERTILITY,
        "feature_importance": {k: round(v, 4) for k, v in fert_importances.items()},
    },
}

with open("metrics.json", "w") as f:
    json.dump(metrics, f, indent=2)

print("Clay model  -> R2: {:.3f}, MAE: {:.2f}".format(clay_r2, clay_mae))
print("Fertility model -> Accuracy: {:.3f}".format(fert_acc))
print("Saved clay_model.pkl, fertility_model.pkl, metrics.json")