"""
==============================================================================
 GWR LOOCV - Estimasi Kemiskinan Kecamatan Sleman
==============================================================================
 Geographically Weighted Regression dengan evaluasi Leave-One-Out CV.
 Alur:
   1. Load data (with centroid lon/lat)
   2. EDA singkat
   3. StandardScaler X (y dibiarkan di skala % asli)
   4. Cari bandwidth optimal via Sel_BW (golden search, AICc)
   5. Fit GWR global -> R2, AICc, koefisien lokal
   6. LOOCV manual: tiap fold, refit GWR dengan 1 titik di-hold-out
   7. Evaluasi RMSE / MAE / MAPE / R2 (LOOCV)
   8. Detail prediksi + simpan CSV
==============================================================================
"""

import warnings
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

from mgwr.gwr import GWR
from mgwr.sel_bw import Sel_BW

from src.modelling.helpers import section, subsection, mape
from src.config import (
    MODEL_DATA_CENTROID_PATH, TARGET_COL, KECAMATAN_COL,
    LONG_COL, LAT_COL, OUTPUT_DIR,
    PRED_COL_AKTUAL, PRED_COL_PREDIKSI, PRED_COL_APE,
)


# ==============================================================================
# STEP 1 - LOAD DATA
# ==============================================================================
section("STEP 1 - LOAD DATA (with centroid)")

df = pd.read_csv(MODEL_DATA_CENTROID_PATH)

# Fix locale-style decimal comma di target
if df[TARGET_COL].dtype == object:
    df[TARGET_COL] = (
        df[TARGET_COL]
        .astype(str)
        .str.replace(",", ".", regex=False)
        .astype(float)
    )

print(f"\n  Jumlah baris (kecamatan) : {len(df)}")
print(f"  Jumlah kolom             : {df.shape[1]}")
print(f"  Kolom centroid           : {LONG_COL}, {LAT_COL}")

# Validasi koordinat
assert LONG_COL in df.columns and LAT_COL in df.columns, (
    f"Kolom centroid '{LONG_COL}'/'{LAT_COL}' tidak ditemukan di {MODEL_DATA_CENTROID_PATH}"
)

print(f"\n  Statistik deskriptif target ({TARGET_COL}):")
print(df[TARGET_COL].describe().to_string())


# ==============================================================================
# STEP 2 - EDA: PEMILIHAN FITUR
# ==============================================================================
section("STEP 2 - PEMILIHAN FITUR")

# Sama dengan loocv_benchmark: drop population_mean (proxy target)
non_feature_cols = [
    KECAMATAN_COL, TARGET_COL,
    LONG_COL, LAT_COL,
    "population_mean",
]
feature_cols = [c for c in df.columns if c not in non_feature_cols]

print(f"\n  Fitur yang digunakan ({len(feature_cols)}):")
for i, f in enumerate(feature_cols, 1):
    print(f"    {i:2}. {f}")
print(f"\n  Target : {TARGET_COL}")
print(f"  Excluded: {non_feature_cols}")


# ==============================================================================
# STEP 3 - PREPROCESSING
# ==============================================================================
section("STEP 3 - PREPROCESSING - StandardScaler (X only)")

X_raw  = df[feature_cols].values
y      = df[TARGET_COL].values.reshape(-1, 1)
coords = list(zip(df[LONG_COL].values, df[LAT_COL].values))

scaler_X = StandardScaler()
X_scaled = scaler_X.fit_transform(X_raw)

print(f"\n  Shape X_scaled : {X_scaled.shape}")
print(f"  Shape y        : {y.shape}")
print(f"  Coords (n)     : {len(coords)}")
print(f"  Mean X_scaled  : {X_scaled.mean(axis=0).round(4)}")
print(f"  Std  X_scaled  : {X_scaled.std(axis=0).round(4)}")
print("  -> y tidak di-scale; GWR fit langsung di skala % aslinya")


# ==============================================================================
# STEP 4 - BANDWIDTH SEARCH
# ==============================================================================
section("STEP 4 - BANDWIDTH SEARCH (AICc, adaptive bisquare)")

# Adaptive bisquare = jumlah tetangga terdekat (k-NN), cocok untuk n kecil
# Constrain search range agar tidak melebihi n (n=17)
n_obs = X_scaled.shape[0]
bw_min_search = max(len(feature_cols) + 2, 5)   # >= jumlah parameter lokal
bw_max_search = n_obs - 1                        # tidak boleh >= n

print(f"\n  Search range bandwidth: [{bw_min_search}, {bw_max_search}]  (n_obs={n_obs})")

sel = Sel_BW(coords, y, X_scaled, fixed=False, kernel="bisquare")
bw_opt = sel.search(
    criterion="AICc",
    bw_min=bw_min_search,
    bw_max=bw_max_search,
)

print(f"\n  Optimal bandwidth (k tetangga): {bw_opt}")
print("  Kernel : adaptive bisquare")
print("  Kriteria : AICc")


# ==============================================================================
# STEP 5 - FIT GWR GLOBAL
# ==============================================================================
section("STEP 5 - FIT GWR GLOBAL (full data)")

gwr_model = GWR(coords, y, X_scaled, bw=bw_opt, fixed=False, kernel="bisquare")
gwr_res   = gwr_model.fit()

print(f"\n  R^2 (in-sample)        : {gwr_res.R2:.4f}")
print(f"  Adj. R^2               : {gwr_res.adj_R2:.4f}")
print(f"  AICc                   : {gwr_res.aicc:.4f}")
print(f"  Effective # parameters : {gwr_res.tr_S:.2f}")
print(f"  Sigma^2                : {gwr_res.sigma2:.4f}")

# Koefisien lokal (n x (k+1)): kolom 0 = intercept
local_coefs = pd.DataFrame(
    gwr_res.params,
    columns=["intercept"] + feature_cols,
)
local_coefs.insert(0, KECAMATAN_COL, df[KECAMATAN_COL].values)

subsection("Ringkasan Koefisien Lokal (mean / std antar lokasi)")
coef_summary = local_coefs.drop(columns=[KECAMATAN_COL]).agg(["mean", "std", "min", "max"]).T
print(coef_summary.round(4).to_string())


# ==============================================================================
# STEP 6 - LOOCV
# ==============================================================================
section("STEP 6 - LOOCV (refit GWR per fold)")

print(f"\n  n fold = {len(y)}  (1 kecamatan hold-out per fold)")
print("  Catatan: bandwidth dipertahankan dari STEP 4 supaya konsisten lintas fold.")
print("           (Re-search bandwidth per fold mahal dan rentan overfit pada n kecil)")

n          = X_scaled.shape[0]
y_true_all = []
y_pred_all = []
kec_names  = df[KECAMATAN_COL].values

# Pastikan bandwidth tidak melebihi ukuran train set (n-1)
bw_loocv = int(min(bw_opt, n - 1))
if bw_loocv != bw_opt:
    print(f"  -> bandwidth di-clamp dari {bw_opt} ke {bw_loocv} agar <= n_train")

for i in range(n):
    train_mask = np.ones(n, dtype=bool)
    train_mask[i] = False

    X_train = X_scaled[train_mask]
    y_train = y[train_mask]
    coords_train = [coords[j] for j in range(n) if j != i]

    X_test  = X_scaled[i:i+1]
    coords_test = [coords[i]]

    # Fit di train
    gwr_fold = GWR(
        coords_train, y_train, X_train,
        bw=bw_loocv, fixed=False, kernel="bisquare",
    )
    gwr_fold_res = gwr_fold.fit()

    # Prediksi pada titik hold-out
    pred_obj = gwr_fold.predict(np.array(coords_test), X_test)
    y_hat = float(pred_obj.predictions[0, 0])

    y_true_all.append(float(y[i, 0]))
    y_pred_all.append(y_hat)

    err = y_hat - y[i, 0]
    print(f"  Fold {i+1:>2} | {kec_names[i]:<15} | "
          f"aktual={y[i,0]:.2f}%  prediksi={y_hat:.2f}%  err={err:+.2f}")

y_true_all = np.array(y_true_all)
y_pred_all = np.array(y_pred_all)


# ==============================================================================
# STEP 7 - METRIK LOOCV
# ==============================================================================
section("STEP 7 - METRIK LOOCV")

rmse_val = float(np.sqrt(mean_squared_error(y_true_all, y_pred_all)))
mae_val  = float(mean_absolute_error(y_true_all, y_pred_all))
mape_val = mape(y_true_all, y_pred_all)
r2_val   = float(r2_score(y_true_all, y_pred_all))

print(f"""
  +---------------------------------------------------------+
  |  MODEL          : GWR (adaptive bisquare, bw={bw_loocv:<3})           |
  |  RMSE  (LOOCV)  : {rmse_val:.4f}                                  |
  |  MAE   (LOOCV)  : {mae_val:.4f}                                  |
  |  MAPE  (LOOCV)  : {mape_val:.2f}%                                 |
  |  R^2   (LOOCV)  : {r2_val:.4f}                                  |
  +---------------------------------------------------------+
""")


# ==============================================================================
# STEP 8 - DETAIL PREDIKSI & SIMPAN
# ==============================================================================
section("STEP 8 - DETAIL PREDIKSI & EXPORT")

subsection("Tabel Prediksi LOOCV")
print(f"\n  {'Kecamatan':<16} {'Aktual (%)':>12} {'Prediksi (%)':>14} {'Error (pp)':>12} {'APE (%)':>10}")
print(f"  {'-'*16} {'-'*12} {'-'*14} {'-'*12} {'-'*10}")
for kec, act, pred in zip(kec_names, y_true_all, y_pred_all):
    err = pred - act
    ape = abs(err / act) * 100 if act != 0 else 0.0
    print(f"  {kec:<16} {act:>12.2f} {pred:>14.2f} {err:>+12.2f} {ape:>9.2f}%")

subsection("Simpan Hasil")

OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
GWR_PRED_PATH  = OUTPUT_DIR / "gwr_loocv_predictions.csv"
GWR_COEF_PATH  = OUTPUT_DIR / "gwr_local_coefficients.csv"

pred_df = pd.DataFrame({
    KECAMATAN_COL    : kec_names,
    PRED_COL_AKTUAL  : y_true_all,
    PRED_COL_PREDIKSI: y_pred_all,
})
pred_df[PRED_COL_APE] = (
    np.abs(pred_df[PRED_COL_PREDIKSI] - pred_df[PRED_COL_AKTUAL])
    / pred_df[PRED_COL_AKTUAL].clip(lower=1e-9) * 100
)
pred_df.to_csv(GWR_PRED_PATH, index=False)
local_coefs.to_csv(GWR_COEF_PATH, index=False)

print(f"  -> Prediksi LOOCV         : {GWR_PRED_PATH}")
print(f"  -> Koefisien lokal (full) : {GWR_COEF_PATH}")

print("\n  [SELESAI] GWR LOOCV pipeline selesai.")