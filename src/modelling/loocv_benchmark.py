"""
==============================================================================
 BARE MODELLING PIPELINE - Estimasi Kemiskinan Kecamatan Kabupaten Sleman
==============================================================================
 Alur kerja:
   1. Load & inspect data
   2. EDA: deskripsi fitur
   3. StandardScaler
   4. LOOCV benchmark: Ridge, Lasso, GPR, SVR, RF, XGBoost
   5. Evaluasi: RMSE, MAE, MAPE
   6. Ringkasan hasil & rekomendasi model terbaik
==============================================================================
"""

import warnings
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import Ridge, Lasso
from sklearn.svm import SVR
from sklearn.ensemble import RandomForestRegressor
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import RBF, WhiteKernel
from sklearn.model_selection import LeaveOneOut
from sklearn.metrics import mean_squared_error, mean_absolute_error
from xgboost import XGBRegressor

from src.modelling.helpers import section, subsection, mape
from src.config import MODEL_DATA_PATH, TARGET_COL, KECAMATAN_COL


# ==============================================================================
# STEP 1 - LOAD DATA
# ==============================================================================
section("STEP 1 - LOAD DATA")

df = pd.read_csv(MODEL_DATA_PATH)

# Fix locale-style decimal comma in target column
if df[TARGET_COL].dtype == object:
    df[TARGET_COL] = (
        df[TARGET_COL]
        .astype(str)
        .str.replace(",", ".", regex=False)
        .astype(float)
    )

print(f"\n  Jumlah baris (kecamatan) : {len(df)}")
print(f"  Jumlah kolom             : {df.shape[1]}")
print(f"\n  Kolom  : {list(df.columns)}")
print(f"\n  Preview (5 baris pertama):")
print(df.head().to_string(index=False))

print(f"\n  Statistik deskriptif target (persentase_penduduk_miskin):")
print(df[TARGET_COL].describe().to_string())


# ==============================================================================
# STEP 2 - EDA: DESKRIPSI FITUR
# ==============================================================================
section("STEP 2 - EDA - DESKRIPSI FITUR")

feature_cols = [c for c in df.columns if c not in [KECAMATAN_COL, TARGET_COL, "population_mean"]]
X_raw = df[feature_cols].copy()
y     = df[TARGET_COL].values

print(f"\n  Menggunakan semua fitur yang tersedia ({len(feature_cols)} fitur):")
for i, f in enumerate(feature_cols, 1):
    print(f"    {i:2}. {f}")

print(f"\n  Target: persentase_penduduk_miskin")


# ==============================================================================
# STEP 3 - PREPROCESSING: STANDARD SCALER
# ==============================================================================
section("STEP 3 - PREPROCESSING - StandardScaler")

X = df[feature_cols].values
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

print(f"\n  Fitur yang di-scale: {feature_cols}")
print(f"  Shape X_scaled     : {X_scaled.shape}")
print(f"  Mean setelah scale : {X_scaled.mean(axis=0).round(6)}")
print(f"  Std setelah scale  : {X_scaled.std(axis=0).round(6)}")
print("  -> Semua fitur sekarang berskala seragam (mean~0, std~1)")


# ==============================================================================
# STEP 4 - LOOCV BENCHMARK
# ==============================================================================
section("STEP 4 - LOOCV BENCHMARK - 6 Model")

print(f"\n  Strategi: Leave-One-Out CV (n={len(y)} fold)")
print("  Setiap fold: 1 kecamatan jadi test, sisanya train")
print("  Metrik: RMSE | MAE | MAPE")

loo = LeaveOneOut()

# -- Definisi model ------------------------------------------------------------
models = {
    "Ridge Regression": Ridge(alpha=1.0),
    "Lasso Regression": Lasso(alpha=0.1, max_iter=10000),
    "Gaussian Process" : GaussianProcessRegressor(
        kernel=RBF(length_scale=1.0) + WhiteKernel(noise_level=0.1),
        n_restarts_optimizer=10,
        random_state=42
    ),
    "SVR (RBF)"       : SVR(kernel="rbf", C=10, epsilon=0.5),
    "Random Forest"   : RandomForestRegressor(
        n_estimators=100, max_depth=3,
        min_samples_leaf=3, random_state=42
    ),
    "XGBoost"         : XGBRegressor(
        n_estimators=50, max_depth=2, learning_rate=0.1,
        reg_lambda=5, reg_alpha=1,
        random_state=42, verbosity=0
    ),
}

results = {}

for name, model in models.items():
    subsection(f"Model: {name}")

    y_true_all, y_pred_all = [], []

    for fold, (train_idx, test_idx) in enumerate(loo.split(X_scaled)):
        X_train, X_test = X_scaled[train_idx], X_scaled[test_idx]
        y_train, y_test = y[train_idx],        y[test_idx]

        model.fit(X_train, y_train)
        pred = model.predict(X_test)[0]

        y_true_all.append(y_test[0])
        y_pred_all.append(pred)

        kec_name = df[KECAMATAN_COL].iloc[test_idx[0]]
        print(f"  Fold {fold+1:>2} | {kec_name:<15} | "
              f"aktual={y_test[0]:.2f}%  prediksi={pred:.2f}%  "
              f"err={abs(y_test[0]-pred):.2f}")

    rmse_val = np.sqrt(mean_squared_error(y_true_all, y_pred_all))
    mae_val  = mean_absolute_error(y_true_all, y_pred_all)
    mape_val = mape(y_true_all, y_pred_all)

    results[name] = {
        "RMSE": rmse_val,
        "MAE" : mae_val,
        "MAPE": mape_val,
        "y_pred": y_pred_all,
    }

    print(f"\n  > RMSE={rmse_val:.4f}  MAE={mae_val:.4f}  MAPE={mape_val:.2f}%")


# ==============================================================================
# STEP 5 - RINGKASAN HASIL & REKOMENDASI
# ==============================================================================
section("STEP 5 - RINGKASAN HASIL LOOCV")

summary = pd.DataFrame({
    name: {
        "RMSE": v["RMSE"],
        "MAE" : v["MAE"],
        "MAPE (%)": v["MAPE"],
    }
    for name, v in results.items()
}).T.sort_values("MAPE (%)")

print(f"\n  {'Model':<22} {'RMSE':>8} {'MAE':>8} {'MAPE (%)':>10}")
print(f"  {'-'*22} {'-'*8} {'-'*8} {'-'*10}")
for i, (name, row) in enumerate(summary.iterrows()):
    marker = "  <- TERBAIK" if i == 0 else ""
    print(f"  {name:<22} {row['RMSE']:>8.4f} {row['MAE']:>8.4f} {row['MAPE (%)']:>9.2f}%{marker}")

best_model_name = summary.index[0]
best_mape       = summary["MAPE (%)"].iloc[0]

print(f"""
  +-----------------------------------------------------+
  |  MODEL TERBAIK  : {best_model_name:<32} |
  |  MAPE (LOOCV)   : {best_mape:.2f}%{' '*38}|
  +-----------------------------------------------------+
""")

# -- Detail prediksi model terbaik --------------------------------------------
subsection(f"Detail Prediksi - {best_model_name}")

best_preds = results[best_model_name]["y_pred"]
print(f"\n  {'Kecamatan':<16} {'Aktual (%)':>12} {'Prediksi (%)':>14} {'Error (pp)':>12}")
print(f"  {'-'*16} {'-'*12} {'-'*14} {'-'*12}")
for i, (kec, act, pred) in enumerate(zip(df[KECAMATAN_COL], y, best_preds)):
    err = pred - act
    print(f"  {kec:<16} {act:>12.2f} {pred:>14.2f} {err:>+12.2f}")

print(f"""
------------------------------------------------------------------
  CATATAN:
  * Error dalam satuan percentage point (pp), bukan persen.
  * MAPE tinggi di dataset kecil sering dipicu outlier satu titik.
  * Cek kecamatan dengan |Error| terbesar untuk validasi lapangan.
  * Jika MAPE masih > 20%, pertimbangkan tambah fitur atau
    gunakan domain knowledge untuk weight kecamatan tertentu.
------------------------------------------------------------------
""")

print("  [SELESAI] Script pipeline selesai dijalankan.")