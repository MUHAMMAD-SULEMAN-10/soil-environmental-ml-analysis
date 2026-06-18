Soil & Environmental Data Analysis using Machine Learning

A machine learning project that models two core soil science problems —
clay content (texture) and soil fertility — from environmental and
chemical properties, wrapped in an interactive Streamlit web app.

🔗 Live demo: add your Streamlit Cloud URL here once deployed
📂 Repo: https://github.com/MUHAMMAD-SULEMAN-10/soil-environmental-ml-analysis


Overview

This project includes two supervised learning models:


Clay Content Regressor — predicts clay percentage (%) from depth,
rainfall, temperature, slope, land use, and parent material. This is
directly relevant to studying lessivage (clay translocation /
illuviation), the process by which clay particles migrate downward
through a soil profile over time and accumulate in subsurface horizons.
Soil Fertility Classifier — classifies soil fertility as Low,
Medium, or High based on organic matter, pH, nitrogen,
phosphorus, potassium, cation exchange capacity (CEC), clay content, and
moisture.


Both models are wrapped in a multi-tab Streamlit interface for exploratory
data analysis, single-sample prediction, batch CSV prediction, and model
diagnostics (feature importance, accuracy, per-class metrics).

Motivation

I built this project to apply machine learning to a domain I'm increasingly
interested in — soil and environmental science — and to practice the full
pipeline end to end: defining the problem, generating/cleaning data,
feature engineering, model training and evaluation, and shipping a usable
interface rather than just a notebook. The clay translocation angle in
particular was chosen because it mirrors real pedological research
questions (how texture changes with depth, climate, and parent material),
even though the data here is synthetic rather than field-collected.

How it was built

1. Data generation (generate_data.py)
Since I didn't have access to a licensed soil survey dataset, I generated
1,500 synthetic samples using equations that encode known pedological
relationships rather than pure random noise, for example:


Clay % increases with depth and annual rainfall (more time and water for
illuviation) and decreases with slope (erosion truncates the
clay-enriched Bt horizon).
Organic matter is higher in wetland and forest land uses, and decreases
with depth and temperature (faster decomposition in warmer soils).
A composite fertility score is built from organic matter, N-P-K, CEC, and
a penalty for pH deviating from the ~6.5 optimum, then z-scored and
bucketed into Low / Medium / High so classes stay reasonably balanced.


This was a deliberate design choice: rather than fabricate arbitrary
numbers, the generator approximates real soil chemistry/physics so the
resulting models learn the kind of relationships you'd actually expect to
see in the field — useful as a sandbox for the modeling pipeline, even
though it's not a substitute for real measurements.

2. Modeling (train_model.py)


Both models use Random Forests (scikit-learn), chosen because
they handle non-linear feature interactions well, need little
preprocessing, and give interpretable feature importances — useful for
understanding which variables actually drive clay content or
fertility.
Categorical variables (land use, parent material) are one-hot encoded
via a ColumnTransformer inside a Pipeline, so preprocessing and
prediction stay bundled together.
Each model is evaluated on a held-out 20% test split (stratified for the
classifier) using R²/MAE for regression and accuracy/F1 for
classification.
Trained models are serialized with joblib, and metrics + feature
importances are saved to metrics.json so the app can load everything
instantly without retraining on every run.


3. Interface (app.py)
Built with Streamlit and Plotly, organized into four tabs:


Data Overview — dataset preview, summary statistics, a correlation
heatmap, and an interactive scatter plot where you can pick any two
features and a color grouping.
Clay Content Prediction — sliders/dropdowns for a single prediction,
plus CSV upload for batch scoring with a downloadable results file.
Fertility Prediction — same pattern, with a probability bar chart
showing the model's confidence across the three classes.
Model Insights — R²/MAE/accuracy/F1, feature importance charts for
both models, and a per-class precision/recall/F1 breakdown.


Skills demonstrated


Data generation and feature engineering grounded in domain logic
Supervised learning (regression + multi-class classification) with
scikit-learn Pipeline/ColumnTransformer
Model evaluation (R², MAE, accuracy, F1, confusion-matrix-style
per-class metrics) and feature importance interpretation
Interactive data visualization (Plotly: heatmaps, histograms, scatter
plots with trendlines, bar charts)
Building and deploying a full-stack data app with Streamlit
Reproducible project structure (separate scripts for data generation,
training, and serving; pinned dependencies; version control)


Honest limitations


The dataset is synthetic, generated from approximate pedological
relationships, not real field measurements — it's a demonstration of the
ML workflow, not a research finding.
The relationships are simplified; real soil systems involve many more
interacting variables (mineralogy, microbial activity, management
history, etc.) than are modeled here.
Model performance (R² ≈ 0.83 for clay, accuracy ≈ 0.75 for fertility)
reflects how well the models recover the synthetic generating function,
not real-world predictive accuracy.


Project structure

soil_ml_project/
├── app.py                 # Streamlit interface (main entry point)

├── generate_data.py       # Builds the synthetic dataset

├── train_model.py         # Trains both models, saves .pkl + metrics.json

├── soil_data.csv          # Generated dataset

├── clay_model.pkl         # Trained Random Forest regressor

├── fertility_model.pkl    # Trained Random Forest classifier

├── metrics.json           # Saved performance metrics + feature importances

└── requirements.txt

Run locally

bashpip install -r requirements.txt
streamlit run app.py

Regenerate data / retrain models (optional)

bashpython generate_data.py   # rebuilds soil_data.csv
python train_model.py     # retrains both models, overwrites .pkl + metrics.json

Deploy on Streamlit Community Cloud


Push this folder to a public GitHub repo (the .pkl files are small
enough — ~15 MB total — to commit directly; no Git LFS needed).
Go to share.streamlit.io and sign in with
GitHub.
Click "New app", select your repo/branch, and set the main file path
to app.py.
Click Deploy. Streamlit Cloud will install requirements.txt
automatically and launch the app — usually ready in 1–2 minutes.
Your app gets a public URL like
https://<your-app-name>.streamlit.app that you can share or link in
a portfolio/CV.


Model performance (on a held-out 20% test split)


Clay regressor: R² ≈ 0.83, MAE ≈ 3.1 percentage points
Fertility classifier: Accuracy ≈ 0.75


(Re-run train_model.py to regenerate these — exact numbers may shift
slightly with each data regeneration since it reseeds randomness.)

Notes on extending this with real data

To swap in real soil data (e.g., from the USDA-NRCS National Cooperative
Soil Survey database), keep the same column names used in
train_model.py, or adjust the NUM_COLS_FOR_CLAY / NUM_COLS_FOR_FERTILITY
lists to match your actual columns, then rerun train_model.py.

Future work


Validate the modeling approach against a real, published soil dataset
(e.g., NCSS Soil Characterization Database)
Add depth-profile visualization (clay % vs. depth for a given soil series)
Experiment with gradient boosting (XGBoost/LightGBM) and compare against
the Random Forest baseline
Add cross-validation and hyperparameter tuning instead of fixed
Random Forest settings
Add unit tests for the data generation and prediction pipeline


Author

Muhammad Suleman
GitHub: @MUHAMMAD-SULEMAN-10

License

This project is open source and available for educational and research use.
