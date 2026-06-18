# Soil & Environmental Data Analysis using Machine Learning

A machine learning project analyzing soil and environmental data, with an
interactive Streamlit interface. Includes two models:

1. **Clay Content Regressor** — predicts clay percentage (%) from depth,
   rainfall, temperature, slope, land use, and parent material. Relevant to
   studying processes like *lessivage* (clay translocation/illuviation).
2. **Soil Fertility Classifier** — classifies soil fertility as Low / Medium
   / High based on organic matter, pH, N-P-K, CEC, clay content, and moisture.

The dataset (`soil_data.csv`, 1,500 samples) is synthetically generated
(`generate_data.py`) using realistic pedological relationships (e.g., clay
accumulates with depth and rainfall, fertility rises with organic matter and
nutrients, falls off as pH strays from ~6.5). It's meant for demonstrating
the ML workflow, not as real field data.

## Project structure

```
soil_ml_project/
├── app.py                 # Streamlit interface (main entry point)
├── generate_data.py       # Builds the synthetic dataset
├── train_model.py         # Trains both models, saves .pkl + metrics.json
├── soil_data.csv          # Generated dataset
├── clay_model.pkl         # Trained Random Forest regressor
├── fertility_model.pkl    # Trained Random Forest classifier
├── metrics.json           # Saved performance metrics + feature importances
└── requirements.txt
```

## Run locally

```bash
pip install -r requirements.txt
streamlit run app.py
```

## Regenerate data / retrain models (optional)

```bash
python generate_data.py   # rebuilds soil_data.csv
python train_model.py     # retrains both models, overwrites .pkl + metrics.json
```

## Deploy on Streamlit Community Cloud

1. Push this folder to a **public GitHub repo** (the `.pkl` files are small
   enough — ~15 MB total — to commit directly; no Git LFS needed).
2. Go to [share.streamlit.io](https://share.streamlit.io) and sign in with
   GitHub.
3. Click **"New app"**, select your repo/branch, and set the main file path
   to `app.py`.
4. Click **Deploy**. Streamlit Cloud will install `requirements.txt`
   automatically and launch the app — usually ready in 1–2 minutes.
5. Your app gets a public URL like
   `https://<your-app-name>.streamlit.app` that you can share or link in
   a portfolio/CV.

## Model performance (on a held-out 20% test split)

- Clay regressor: R² ≈ 0.83, MAE ≈ 3.1 percentage points
- Fertility classifier: Accuracy ≈ 0.75

(Re-run `train_model.py` to regenerate these — exact numbers may shift
slightly with each data regeneration since it reseeds randomness.)

## Notes on extending this with real data

To swap in real soil data (e.g., from the USDA-NRCS National Cooperative
Soil Survey database), keep the same column names used in
`train_model.py`, or adjust the `NUM_COLS_FOR_CLAY` / `NUM_COLS_FOR_FERTILITY`
lists to match your actual columns, then rerun `train_model.py`.
