"""
app.py
Streamlit interface for the Soil & Environmental Data Analysis ML project.

Run locally with:
    streamlit run app.py

Deploy on Streamlit Community Cloud by pointing it at this file in your repo.
"""

import json

import joblib
import numpy as np
import pandas as pd
import plotly.express as px
import streamlit as st

st.set_page_config(
    page_title="Soil & Environmental ML Analysis",
    page_icon="🌱",
    layout="wide",
)

# ---------------------------------------------------------------
# Cached loaders
# ---------------------------------------------------------------
@st.cache_data
def load_data():
    return pd.read_csv("soil_data.csv")


@st.cache_resource
def load_models():
    clay_model = joblib.load("clay_model.pkl")
    fert_model = joblib.load("fertility_model.pkl")
    with open("metrics.json") as f:
        metrics = json.load(f)
    return clay_model, fert_model, metrics


df = load_data()
clay_model, fert_model, metrics = load_models()

CLASS_COLORS = {"Low": "#d62728", "Medium": "#f0a500", "High": "#2ca02c"}

# ---------------------------------------------------------------
# Sidebar
# ---------------------------------------------------------------
with st.sidebar:
    st.title("🌱 Soil ML Explorer")
    st.markdown(
        "A machine learning project on soil & environmental data, covering:\n\n"
        "- **Clay content prediction** from depth, climate, and parent material "
        "(relevant to clay translocation / lessivage studies)\n"
        "- **Soil fertility classification** from chemical properties\n\n"
        "Built with `scikit-learn` + `Streamlit`."
    )
    st.markdown("---")
    st.caption(f"Dataset: {df.shape[0]} samples · {df.shape[1]} features")
    st.caption("Models: Random Forest (regression + classification)")

st.title("Soil & Environmental Data Analysis using Machine Learning")
st.markdown(
    "Explore a synthetic but pedologically-informed soil dataset, predict "
    "**clay content** and **fertility class** from site conditions, and "
    "inspect what the models learned."
)

tab_overview, tab_clay, tab_fert, tab_models = st.tabs(
    ["📊 Data Overview", "🧱 Clay Content Prediction", "🌾 Fertility Prediction", "📈 Model Insights"]
)

# ---------------------------------------------------------------
# TAB 1: Data Overview / EDA
# ---------------------------------------------------------------
with tab_overview:
    st.subheader("Dataset Preview")
    st.dataframe(df.head(20), use_container_width=True)

    col1, col2 = st.columns(2)
    with col1:
        st.subheader("Summary Statistics")
        st.dataframe(df.describe().T.round(2), use_container_width=True)

    with col2:
        st.subheader("Fertility Class Distribution")
        fig = px.histogram(
            df, x="Fertility_Class", color="Fertility_Class",
            color_discrete_map=CLASS_COLORS,
            category_orders={"Fertility_Class": ["Low", "Medium", "High"]},
        )
        fig.update_layout(showlegend=False)
        st.plotly_chart(fig, use_container_width=True)

    st.subheader("Feature Correlation Heatmap")
    numeric_df = df.select_dtypes(include=np.number)
    corr = numeric_df.corr()
    fig_corr = px.imshow(
        corr, text_auto=".2f", aspect="auto",
        color_continuous_scale="RdBu_r", zmin=-1, zmax=1,
    )
    st.plotly_chart(fig_corr, use_container_width=True)

    st.subheader("Explore Relationships")
    c1, c2, c3 = st.columns(3)
    with c1:
        x_axis = st.selectbox("X-axis", numeric_df.columns, index=list(numeric_df.columns).index("Depth_cm"))
    with c2:
        y_axis = st.selectbox("Y-axis", numeric_df.columns, index=list(numeric_df.columns).index("Clay_pct"))
    with c3:
        color_by = st.selectbox("Color by", ["Fertility_Class", "Land_Use", "Parent_Material"])

    fig_scatter = px.scatter(
        df, x=x_axis, y=y_axis, color=color_by,
        color_discrete_map=CLASS_COLORS if color_by == "Fertility_Class" else None,
        opacity=0.7, trendline="ols",
    )
    st.plotly_chart(fig_scatter, use_container_width=True)

# ---------------------------------------------------------------
# TAB 2: Clay content prediction
# ---------------------------------------------------------------
with tab_clay:
    st.subheader("Predict Clay Content (%)")
    st.markdown(
        "This model estimates clay percentage from depth and site conditions — "
        "useful for studying processes like **lessivage / clay translocation**, "
        "where clay accumulates in subsurface horizons over time."
    )

    c1, c2 = st.columns(2)
    with c1:
        depth = st.slider("Depth (cm)", 0.0, 150.0, 50.0)
        rainfall = st.slider("Annual Rainfall (mm)", 200.0, 2200.0, 950.0)
        temperature = st.slider("Mean Annual Temperature (°C)", 2.0, 32.0, 18.0)
    with c2:
        slope = st.slider("Slope (%)", 0.0, 45.0, 5.0)
        land_use = st.selectbox("Land Use", df["Land_Use"].unique())
        parent_material = st.selectbox("Parent Material", df["Parent_Material"].unique())

    input_df = pd.DataFrame([{
        "Depth_cm": depth,
        "Rainfall_mm": rainfall,
        "Temperature_C": temperature,
        "Slope_pct": slope,
        "Land_Use": land_use,
        "Parent_Material": parent_material,
    }])

    if st.button("Predict Clay Content", type="primary"):
        pred = clay_model.predict(input_df)[0]
        st.metric("Predicted Clay Content", f"{pred:.1f}%")

        # Simple texture interpretation
        if pred < 15:
            texture_note = "Sandy-dominant texture — expect high drainage, low water retention."
        elif pred < 35:
            texture_note = "Loamy texture — balanced drainage and nutrient retention."
        else:
            texture_note = "Clay-dominant texture — high water/nutrient retention, slower drainage."
        st.info(texture_note)

    st.markdown("---")
    st.subheader("Batch Prediction (CSV upload)")
    st.caption(
        "Upload a CSV with columns: Depth_cm, Rainfall_mm, Temperature_C, "
        "Slope_pct, Land_Use, Parent_Material"
    )
    uploaded = st.file_uploader("Upload CSV", type="csv", key="clay_upload")
    if uploaded is not None:
        batch_df = pd.read_csv(uploaded)
        try:
            batch_df["Predicted_Clay_pct"] = clay_model.predict(batch_df)
            st.dataframe(batch_df, use_container_width=True)
            st.download_button(
                "Download Predictions",
                batch_df.to_csv(index=False),
                "clay_predictions.csv",
                "text/csv",
            )
        except Exception as e:
            st.error(f"Could not score file — check required columns. Details: {e}")

# ---------------------------------------------------------------
# TAB 3: Fertility classification
# ---------------------------------------------------------------
with tab_fert:
    st.subheader("Predict Soil Fertility Class")
    st.markdown("Classify soil fertility as **Low**, **Medium**, or **High** based on chemical properties.")

    c1, c2, c3 = st.columns(3)
    with c1:
        om = st.slider("Organic Matter (%)", 0.2, 12.0, 4.0)
        ph = st.slider("pH", 3.8, 8.5, 6.5)
        moisture = st.slider("Moisture (%)", 2.0, 55.0, 20.0)
    with c2:
        nitrogen = st.slider("Nitrogen (mg/kg)", 20.0, 1200.0, 400.0)
        phosphorus = st.slider("Phosphorus (mg/kg)", 2.0, 120.0, 40.0)
    with c3:
        potassium = st.slider("Potassium (mg/kg)", 20.0, 600.0, 150.0)
        cec = st.slider("CEC (cmol/kg)", 2.0, 45.0, 15.0)
        clay = st.slider("Clay (%)", 2.0, 65.0, 25.0)

    fert_input = pd.DataFrame([{
        "Organic_Matter_pct": om,
        "pH": ph,
        "Nitrogen_mgkg": nitrogen,
        "Phosphorus_mgkg": phosphorus,
        "Potassium_mgkg": potassium,
        "CEC_cmolkg": cec,
        "Clay_pct": clay,
        "Moisture_pct": moisture,
    }])

    if st.button("Predict Fertility Class", type="primary"):
        pred_class = fert_model.predict(fert_input)[0]
        proba = fert_model.predict_proba(fert_input)[0]
        classes = fert_model.named_steps["model"].classes_

        color = CLASS_COLORS.get(pred_class, "#444")
        st.markdown(
            f"### Predicted Class: <span style='color:{color}'>{pred_class}</span>",
            unsafe_allow_html=True,
        )

        proba_df = pd.DataFrame({"Class": classes, "Probability": proba})
        fig = px.bar(
            proba_df, x="Class", y="Probability", color="Class",
            color_discrete_map=CLASS_COLORS, range_y=[0, 1],
        )
        fig.update_layout(showlegend=False)
        st.plotly_chart(fig, use_container_width=True)

    st.markdown("---")
    st.subheader("Batch Prediction (CSV upload)")
    st.caption(
        "Upload a CSV with columns: Organic_Matter_pct, pH, Nitrogen_mgkg, "
        "Phosphorus_mgkg, Potassium_mgkg, CEC_cmolkg, Clay_pct, Moisture_pct"
    )
    uploaded_f = st.file_uploader("Upload CSV", type="csv", key="fert_upload")
    if uploaded_f is not None:
        batch_f = pd.read_csv(uploaded_f)
        try:
            batch_f["Predicted_Fertility_Class"] = fert_model.predict(batch_f)
            st.dataframe(batch_f, use_container_width=True)
            st.download_button(
                "Download Predictions",
                batch_f.to_csv(index=False),
                "fertility_predictions.csv",
                "text/csv",
            )
        except Exception as e:
            st.error(f"Could not score file — check required columns. Details: {e}")

# ---------------------------------------------------------------
# TAB 4: Model insights
# ---------------------------------------------------------------
with tab_models:
    st.subheader("Model Performance")
    c1, c2 = st.columns(2)
    with c1:
        st.markdown("**Clay Content Regressor** (Random Forest)")
        st.metric("R² Score", metrics["clay_model"]["r2"])
        st.metric("Mean Absolute Error", f"{metrics['clay_model']['mae']} pts")
    with c2:
        st.markdown("**Fertility Classifier** (Random Forest)")
        st.metric("Accuracy", metrics["fertility_model"]["accuracy"])
        report = metrics["fertility_model"]["classification_report"]
        st.metric("Macro F1-score", round(report["macro avg"]["f1-score"], 3))

    st.markdown("---")
    c3, c4 = st.columns(2)
    with c3:
        st.subheader("Clay Model — Feature Importance")
        fi = pd.DataFrame(
            list(metrics["clay_model"]["feature_importance"].items()),
            columns=["Feature", "Importance"],
        ).sort_values("Importance", ascending=True)
        fig_fi1 = px.bar(fi, x="Importance", y="Feature", orientation="h")
        st.plotly_chart(fig_fi1, use_container_width=True)

    with c4:
        st.subheader("Fertility Model — Feature Importance")
        fi2 = pd.DataFrame(
            list(metrics["fertility_model"]["feature_importance"].items()),
            columns=["Feature", "Importance"],
        ).sort_values("Importance", ascending=True)
        fig_fi2 = px.bar(fi2, x="Importance", y="Feature", orientation="h")
        st.plotly_chart(fig_fi2, use_container_width=True)

    st.markdown("---")
    st.subheader("Fertility Classifier — Per-class Metrics")
    rows = []
    for cls in ["Low", "Medium", "High"]:
        if cls in report:
            rows.append({
                "Class": cls,
                "Precision": round(report[cls]["precision"], 3),
                "Recall": round(report[cls]["recall"], 3),
                "F1-score": round(report[cls]["f1-score"], 3),
                "Support": int(report[cls]["support"]),
            })
    st.dataframe(pd.DataFrame(rows), use_container_width=True, hide_index=True)

st.markdown("---")
st.caption(
    "Note: this dataset is synthetically generated with realistic pedological "
    "relationships for demonstration purposes, not real field measurements."
)