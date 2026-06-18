"""
generate_data.py
Generates a realistic synthetic soil & environmental dataset.

Features are built with real pedological relationships in mind:
- Clay content increases with depth (illuviation/lessivage), rainfall, and
  decreases with slope (erosion truncates the Bt horizon).
- Soil fertility depends on organic matter, N-P-K, pH (optimal near 6.5),
  and CEC (cation exchange capacity), which itself depends on clay + OM.
"""

import numpy as np
import pandas as pd

np.random.seed(42)
N = 1500

# --- Base environmental / site variables ---
depth_cm = np.random.uniform(0, 150, N)
rainfall_mm = np.random.normal(950, 250, N).clip(200, 2200)
temperature_c = np.random.normal(18, 6, N).clip(2, 32)
slope_pct = np.random.exponential(5, N).clip(0, 45)
land_use = np.random.choice(
    ["Forest", "Cropland", "Grassland", "Wetland"], N, p=[0.3, 0.35, 0.25, 0.10]
)
parent_material = np.random.choice(
    ["Alluvium", "Residuum", "Loess", "Glacial Till"], N, p=[0.3, 0.3, 0.2, 0.2]
)

# --- Soil texture: clay % increases with depth & rainfall (lessivage),
#     decreases with slope (erosion) ---
clay_pct = (
    8 + 0.18 * depth_cm + 0.012 * rainfall_mm - 0.35 * slope_pct
    + np.random.normal(0, 4, N)
).clip(2, 65)

sand_pct = (70 - 0.55 * clay_pct + np.random.normal(0, 6, N)).clip(5, 90)
silt_pct = (100 - clay_pct - sand_pct).clip(2, 80)
total = clay_pct + sand_pct + silt_pct
clay_pct, sand_pct, silt_pct = (clay_pct / total * 100, sand_pct / total * 100, silt_pct / total * 100)

# --- Chemical / fertility properties ---
organic_matter_pct = (
    6.0 - 0.025 * depth_cm + 0.002 * rainfall_mm - 0.04 * temperature_c
    + np.where(land_use == "Wetland", 2.5, 0)
    + np.where(land_use == "Forest", 0.8, 0)
    + np.random.normal(0, 0.8, N)
).clip(0.2, 12)

ph = (
    6.6 - 0.0015 * rainfall_mm
    + np.where(parent_material == "Loess", 0.3, 0)
    + np.where(parent_material == "Glacial Till", 0.2, 0)
    + np.random.normal(0, 0.5, N)
).clip(3.8, 8.5)

nitrogen_mgkg = (organic_matter_pct * 95 + np.random.normal(0, 60, N)).clip(20, 1200)
phosphorus_mgkg = (
    15 + 1.8 * organic_matter_pct - 0.05 * np.abs(ph - 6.5) * 40 + np.random.normal(0, 8, N)
).clip(2, 120)
potassium_mgkg = (
    60 + 2.2 * clay_pct + 1.5 * organic_matter_pct + np.random.normal(0, 25, N)
).clip(20, 600)

moisture_pct = (
    8 + 0.18 * clay_pct + 0.01 * rainfall_mm - 0.15 * sand_pct + np.random.normal(0, 3, N)
).clip(2, 55)

cec = (0.45 * clay_pct + 2.1 * organic_matter_pct + np.random.normal(0, 2, N)).clip(2, 45)

# --- Soil Fertility Index (0-100) ---
# Build a raw composite score, then rescale to a 0-100 distribution
# centered around 50 so Low/Medium/High classes come out reasonably balanced.
ph_penalty = np.abs(ph - 6.5) * 8
raw_score = (
    3.2 * organic_matter_pct + 0.04 * nitrogen_mgkg + 0.25 * phosphorus_mgkg
    + 0.05 * potassium_mgkg + 0.6 * cec - ph_penalty
    + np.random.normal(0, 6, N)
)
z = (raw_score - raw_score.mean()) / raw_score.std()
fertility_score = (50 + 15 * z).clip(0, 100)

fertility_class = pd.cut(
    fertility_score,
    bins=[-1, 40, 60, 101],
    labels=["Low", "Medium", "High"],
)

df = pd.DataFrame({
    "Depth_cm": depth_cm.round(1),
    "Rainfall_mm": rainfall_mm.round(1),
    "Temperature_C": temperature_c.round(1),
    "Slope_pct": slope_pct.round(2),
    "Land_Use": land_use,
    "Parent_Material": parent_material,
    "Clay_pct": clay_pct.round(2),
    "Sand_pct": sand_pct.round(2),
    "Silt_pct": silt_pct.round(2),
    "Organic_Matter_pct": organic_matter_pct.round(2),
    "pH": ph.round(2),
    "Nitrogen_mgkg": nitrogen_mgkg.round(1),
    "Phosphorus_mgkg": phosphorus_mgkg.round(1),
    "Potassium_mgkg": potassium_mgkg.round(1),
    "Moisture_pct": moisture_pct.round(2),
    "CEC_cmolkg": cec.round(2),
    "Fertility_Score": fertility_score.round(1),
    "Fertility_Class": fertility_class,
})

df.to_csv("soil_data.csv", index=False)
print("Saved soil_data.csv with shape:", df.shape)
print(df.head())
print("\nFertility class distribution:\n", df["Fertility_Class"].value_counts())