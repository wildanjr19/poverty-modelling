"""
==============================================================================
 SAR LOOCV - Estimasi Kemiskinan Kecamatan Sleman
==============================================================================
 Spatial Autoregressive Models untuk data kecamatan dengan spatial weights.
 Membandingkan 3 spesifikasi:
   - OLS  : baseline non-spatial
   - SAR  : Spatial Lag Model    -> y = rho*W*y + X*beta + e
   - SEM  : Spatial Error Model  -> y = X*beta + u, u = lambda*W*u + e

 Alur:
   1. Load data (with centroid lon/lat)
   2. Pemilihan fitur (12 fitur, exclude population_mean)
   3. StandardScaler X (y di-scale juga supaya ML_Lag stabil)
   4. Bangun spatial weights matrix (KNN, k=3, row-standardized)
   5. Diagnostik spasial awal: Moran's I pada y
   6. Fit OLS / SAR / SEM full sample -> ringkasan koefisien & rho/lambda
   7. LOOCV manual ketiga model -> RMSE / MAE / MAPE / R^2
   8. Detail prediksi model terbaik + simpan CSV
==============================================================================
"""

import warnings
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

from libpysal.weights import KNN
from esda.moran import Moran
from spreg import OLS, ML_Lag, ML_Error

from src.modelling.helpers import section, subsection, mape
from src.config import (
    MODEL_DATA_CENTROID_PATH, TARGET_COL, KECAMATAN_COL,
    LONG_COL, LAT_COL, OUTPUT_DIR,
    PRED_COL_AKTUAL, PRED_COL_PREDIKSI, PRED_COL_APE,
    MORAN_K_NEIGHBORS, MORAN_PERMUTATIONS,
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

assert LONG_COL in df.columns and LAT_COL in df.columns, (
    f"Kolom centroid '{LONG_COL}'/'{LAT_COL}' tidak ditemukan di {MODEL_DATA_CENTROID_PATH}"
)

print(f"\n  Statistik deskriptif target ({TARGET_COL}):")
print(df[TARGET_COL].describe().to_string())


# ==============================================================================
# STEP 2 - PEMILIHAN FITUR
# ==============================================================================
section("STEP 2 - PEMILIHAN FITUR")

non_feature_cols = [
    KECAMATAN_COL, TARGET_COL,
    LONG_COL, LAT_COL,
    "population_mean",
]
feature_cols = [c for c in df.columns if c not in non_feature_cols]

print(f"\n  Fitur yang digunakan ({len(feature_cols)}):")
for i, f in enumerate(feature_cols, 1):
    print(f"    {i:2}. {f}")
print(f"\n  Target  : {TARGET_COL}")
print(f"  Excluded: {non_feature_cols}")


# ==============================================================================
# STEP 3 - PREPROCESSING
# ==============================================================================
section("STEP 3 - PREPROCESSING - StandardScaler X dan y")

X_raw  = df[feature_cols].values
y_raw  = df[TARGET_COL].values.reshape(-1, 1)
coords = np.column_stack([df[LONG_COL].values, df[LAT_COL].values])

scaler_X = StandardScaler()
scaler_y = StandardScaler()

X_scaled = scaler_X.fit_transform(X_raw)
y_scaled = scaler_y.fit_transform(y_raw)

print(f"\n  Shape X_scaled : {X_scaled.shape}")
print(f"  Shape y_scaled : {y_scaled.shape}")
print(f"  Coords shape   : {coords.shape}")
print("  -> y di-scale supaya ML_Lag/ML_Error konvergen stabil; di-inverse saat metrik")


# ==============================================================================
# STEP 4 - SPATIAL WEIGHTS MATRIX
# ==============================================================================
section(f"STEP 4 - SPATIAL WEIGHTS (KNN k={MORAN_K_NEIGHBORS}, row-standardized)")

w_full = KNN.from_array(coords, k=MORAN_K_NEIGHBORS)
w_full.transform = "r"

print(f"\n  n             : {w_full.n}")
print(f"  k tetangga    : {MORAN_K_NEIGHBORS}")
print(f"  pct_nonzero   : {w_full.pct_nonzero:.2f}%")
print(f"  mean_neighbors: {w_full.mean_neighbors:.2f}")


# ==============================================================================
# STEP 5 - DIAGNOSTIK SPASIAL: MORAN'S I PADA TARGET
# ==============================================================================
section("STEP 5 - MORAN'S I (target asli)")

moran_y = Moran(y_raw.ravel(), w_full, permutations=MORAN_PERMUTATIONS)
print(f"\n  Moran's I       : {moran_y.I:.4f}")
print(f"  Expected I      : {moran_y.EI:.4f}")
print(f"  p-value (sim)   : {moran_y.p_sim:.4f}")
print(f"  z-score (sim)   : {moran_y.z_sim:.4f}")
if moran_y.p_sim < 0.05:
    sign = "POSITIF" if moran_y.I > moran_y.EI else "NEGATIF"
    print(f"  -> Autokorelasi spasial {sign} signifikan (alpha=0.05)")
    print("     SAR/SEM punya justifikasi statistik di sini.")
else:
    print("  -> Tidak ada bukti kuat autokorelasi spasial pada target;")
    print("     SAR/SEM mungkin tidak banyak menambah nilai vs OLS.")


# ==============================================================================
# STEP 6 - FIT MODEL FULL SAMPLE
# ==============================================================================
section("STEP 6 - FIT OLS / SAR / SEM (full sample)")

subsection("OLS baseline")
ols_full = OLS(
    y=y_scaled, x=X_scaled,
    name_y=TARGET_COL, name_x=feature_cols,
    spat_diag=True, w=w_full,
)
print(f"  R^2 (in-sample)   : {ols_full.r2:.4f}")
print(f"  AIC               : {ols_full.aic:.4f}")
print(f"  Schwarz (BIC)     : {ols_full.schwarz:.4f}")
# Residual Moran's I via OLS spatial diagnostics
if hasattr(ols_full, "moran_res") and ols_full.moran_res is not None:
    mi_val, mi_z, mi_p = ols_full.moran_res
    print(f"  Moran's I (resid) : {mi_val:.4f}  (p={mi_p:.4f})")
# Lagrange multiplier tests untuk membantu pilih SAR vs SEM
if hasattr(ols_full, "lm_lag") and ols_full.lm_lag is not None:
    print(f"  LM-lag            : stat={ols_full.lm_lag[0]:.4f}  p={ols_full.lm_lag[1]:.4f}")
if hasattr(ols_full, "lm_error") and ols_full.lm_error is not None:
    print(f"  LM-error          : stat={ols_full.lm_error[0]:.4f}  p={ols_full.lm_error[1]:.4f}")
if hasattr(ols_full, "rlm_lag") and ols_full.rlm_lag is not None:
    print(f"  Robust LM-lag     : stat={ols_full.rlm_lag[0]:.4f}  p={ols_full.rlm_lag[1]:.4f}")
if hasattr(ols_full, "rlm_error") and ols_full.rlm_error is not None:
    print(f"  Robust LM-error   : stat={ols_full.rlm_error[0]:.4f}  p={ols_full.rlm_error[1]:.4f}")

subsection("SAR (Spatial Lag, ML)")
sar_full = ML_Lag(
    y=y_scaled, x=X_scaled, w=w_full,
    name_y=TARGET_COL, name_x=feature_cols,
)
print(f"  rho               : {float(sar_full.rho):.4f}")
print(f"  Pseudo R^2        : {sar_full.pr2:.4f}")
print(f"  AIC               : {sar_full.aic:.4f}")
print(f"  Schwarz (BIC)     : {sar_full.schwarz:.4f}")
print(f"  Log-likelihood    : {sar_full.logll:.4f}")

subsection("SEM (Spatial Error, ML)")
sem_full = ML_Error(
    y=y_scaled, x=X_scaled, w=w_full,
    name_y=TARGET_COL, name_x=feature_cols,
)
print(f"  lambda            : {float(sem_full.lam):.4f}")
print(f"  Pseudo R^2        : {sem_full.pr2:.4f}")
print(f"  AIC               : {sem_full.aic:.4f}")
print(f"  Schwarz (BIC)     : {sem_full.schwarz:.4f}")
print(f"  Log-likelihood    : {sem_full.logll:.4f}")

subsection("Ringkasan koefisien (skala fitur ter-standar)")
coef_table = pd.DataFrame({
    "feature": ["intercept"] + feature_cols,
    "OLS_beta":  ols_full.betas.ravel(),
    "SAR_beta":  sar_full.betas.ravel()[: len(feature_cols) + 1],
    "SEM_beta":  sem_full.betas.ravel()[: len(feature_cols) + 1],
})
print(coef_table.round(4).to_string(index=False))


# ==============================================================================
# STEP 7 - LOOCV (3 model)
# ==============================================================================
section("STEP 7 - LOOCV (OLS / SAR / SEM)")

print(f"\n  n fold = {len(y_scaled)} (1 kecamatan hold-out per fold)")
print("  Strategi prediksi titik hold-out:")
print("    OLS : y_hat = X_test @ beta")
print("    SAR : reduced form -> y_hat dari (I - rho*W_full)^-1 ( X*beta ) pada baris i")
print("          (refit pada train, evaluasi di baris i menggunakan W_full)")
print("    SEM : y_hat = X_test @ beta  (lambda hanya di error term)")

n         = X_scaled.shape[0]
kec_names = df[KECAMATAN_COL].values

def _inv_y(y_std_scalar: float) -> float:
    """Inverse transform ke skala % asli."""
    return float(scaler_y.inverse_transform(np.array([[y_std_scalar]]))[0, 0])

def predict_sar_loocv(X_full_scaled, y_train_full_scaled, sar_res, w_full_obj, test_idx):
    """
    Reduced-form prediction untuk titik hold-out i:
      y_full = (I - rho*W)^-1 * (X*beta + intercept)
    Pakai X_full + W_full, tapi koefisien dari model yg di-fit di train (n-1 obs).
    Untuk W_full kita gunakan baris i saja sebagai jembatan tetangga ke titik i.
    """
    rho = float(sar_res.rho)
    betas = sar_res.betas.ravel()[: X_full_scaled.shape[1] + 1]  # intercept + p
    # X_full dengan intercept
    X_full_const = np.hstack([np.ones((X_full_scaled.shape[0], 1)), X_full_scaled])
    Xb = X_full_const @ betas
    W = w_full_obj.full()[0]  # dense matrix n x n
    I = np.eye(W.shape[0])
    A = I - rho * W
    # Solve untuk full y_hat
    y_full_hat = np.linalg.solve(A, Xb)
    return float(y_full_hat[test_idx])

def predict_ols_or_sem(X_test_scaled, betas):
    """y_hat = [1, x] @ beta untuk OLS dan SEM (lambda di error, tidak masuk mean)."""
    X_const = np.hstack([np.ones((X_test_scaled.shape[0], 1)), X_test_scaled])
    return float((X_const @ betas)[0])

results = {
    "OLS": {"y_pred": [], "y_true": []},
    "SAR": {"y_pred": [], "y_true": []},
    "SEM": {"y_pred": [], "y_true": []},
}

for i in range(n):
    train_mask = np.ones(n, dtype=bool)
    train_mask[i] = False

    X_train = X_scaled[train_mask]
    y_train = y_scaled[train_mask]
    X_test  = X_scaled[i:i+1]
    coords_train = coords[train_mask]

    # Spatial weights di training set saja (untuk fit SAR/SEM)
    w_train = KNN.from_array(coords_train, k=MORAN_K_NEIGHBORS)
    w_train.transform = "r"

    # -- OLS --
    try:
        ols_f = OLS(y=y_train, x=X_train,
                    name_y=TARGET_COL, name_x=feature_cols)
        y_hat_ols = predict_ols_or_sem(X_test, ols_f.betas.ravel())
    except Exception as e:
        print(f"  Fold {i+1} OLS error: {e}")
        y_hat_ols = np.nan

    # -- SAR --
    try:
        sar_f = ML_Lag(y=y_train, x=X_train, w=w_train,
                       name_y=TARGET_COL, name_x=feature_cols)
        y_hat_sar = predict_sar_loocv(X_scaled, y_scaled, sar_f, w_full, i)
    except Exception as e:
        print(f"  Fold {i+1} SAR error: {e}")
        y_hat_sar = np.nan

    # -- SEM --
    try:
        sem_f = ML_Error(y=y_train, x=X_train, w=w_train,
                         name_y=TARGET_COL, name_x=feature_cols)
        # ML_Error.betas berbentuk (p+2, 1): intercept + p coef + lambda
        # Slice untuk ambil intercept + p coef saja
        sem_betas = sem_f.betas.ravel()[: X_train.shape[1] + 1]
        y_hat_sem = predict_ols_or_sem(X_test, sem_betas)
    except Exception as e:
        print(f"  Fold {i+1} SEM error: {e}")
        y_hat_sem = np.nan

    # Inverse scale ke skala asli
    y_true_orig = _inv_y(float(y_scaled[i, 0]))
    y_hat_ols_o = _inv_y(y_hat_ols) if not np.isnan(y_hat_ols) else np.nan
    y_hat_sar_o = _inv_y(y_hat_sar) if not np.isnan(y_hat_sar) else np.nan
    y_hat_sem_o = _inv_y(y_hat_sem) if not np.isnan(y_hat_sem) else np.nan

    for name, val in zip(["OLS", "SAR", "SEM"],
                         [y_hat_ols_o, y_hat_sar_o, y_hat_sem_o]):
        results[name]["y_true"].append(y_true_orig)
        results[name]["y_pred"].append(val)

    print(f"  Fold {i+1:>2} | {kec_names[i]:<15} | "
          f"aktual={y_true_orig:.4f}  "
          f"OLS={y_hat_ols_o:.4f}  SAR={y_hat_sar_o:.4f}  SEM={y_hat_sem_o:.4f}")


# ==============================================================================
# STEP 8 - METRIK LOOCV
# ==============================================================================
section("STEP 8 - METRIK LOOCV (skala % asli)")

def _metrics(y_true, y_pred):
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)
    mask = ~(np.isnan(y_true) | np.isnan(y_pred))
    yt, yp = y_true[mask], y_pred[mask]
    return {
        "n_valid": int(mask.sum()),
        "RMSE": float(np.sqrt(mean_squared_error(yt, yp))),
        "MAE":  float(mean_absolute_error(yt, yp)),
        "MAPE": mape(yt, yp),
        "R2":   float(r2_score(yt, yp)),
    }

summary_rows = []
for name in ["OLS", "SAR", "SEM"]:
    m = _metrics(results[name]["y_true"], results[name]["y_pred"])
    summary_rows.append({"model": name, **m})

summary_df = pd.DataFrame(summary_rows)
summary_df = summary_df.sort_values("MAPE").reset_index(drop=True)

print(f"\n  {'Model':<6} {'n':>4} {'RMSE':>8} {'MAE':>8} {'MAPE':>9} {'R2':>8}")
print(f"  {'-'*6} {'-'*4} {'-'*8} {'-'*8} {'-'*9} {'-'*8}")
for i, r in summary_df.iterrows():
    marker = "  <- TERBAIK" if i == 0 else ""
    print(f"  {r['model']:<6} {r['n_valid']:>4d} "
          f"{r['RMSE']:>8.4f} {r['MAE']:>8.4f} "
          f"{r['MAPE']:>8.2f}% {r['R2']:>8.4f}{marker}")

best_name = summary_df.iloc[0]["model"]
best_row  = summary_df.iloc[0]
print(f"""
  +---------------------------------------------------------+
  |  MODEL TERBAIK   : {best_name:<37} |
  |  RMSE  (LOOCV)   : {best_row['RMSE']:.4f}                                  |
  |  MAE   (LOOCV)   : {best_row['MAE']:.4f}                                  |
  |  MAPE  (LOOCV)   : {best_row['MAPE']:.2f}%                                 |
  |  R^2   (LOOCV)   : {best_row['R2']:.4f}                                  |
  +---------------------------------------------------------+
""")


# ==============================================================================
# STEP 9 - DETAIL PREDIKSI MODEL TERBAIK + EXPORT
# ==============================================================================
section(f"STEP 9 - DETAIL PREDIKSI - {best_name}")

best_pred = np.array(results[best_name]["y_pred"])
best_true = np.array(results[best_name]["y_true"])

print(f"\n  {'Kecamatan':<16} {'Aktual':>10} {'Prediksi':>10} {'Error':>10} {'APE (%)':>10}")
print(f"  {'-'*16} {'-'*10} {'-'*10} {'-'*10} {'-'*10}")
for kec, act, pred in zip(kec_names, best_true, best_pred):
    err = pred - act
    ape = abs(err / act) * 100 if act != 0 else 0.0
    print(f"  {kec:<16} {act:>10.4f} {pred:>10.4f} {err:>+10.4f} {ape:>9.2f}%")

subsection("Simpan Hasil")

OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
SAR_PRED_PATH    = OUTPUT_DIR / "sar_loocv_predictions.csv"
SAR_SUMMARY_PATH = OUTPUT_DIR / "sar_loocv_summary.csv"

pred_df = pd.DataFrame({
    KECAMATAN_COL    : kec_names,
    PRED_COL_AKTUAL  : best_true,
    f"OLS_{PRED_COL_PREDIKSI}": results["OLS"]["y_pred"],
    f"SAR_{PRED_COL_PREDIKSI}": results["SAR"]["y_pred"],
    f"SEM_{PRED_COL_PREDIKSI}": results["SEM"]["y_pred"],
    PRED_COL_PREDIKSI: best_pred,
})
pred_df[PRED_COL_APE] = (
    np.abs(pred_df[PRED_COL_PREDIKSI] - pred_df[PRED_COL_AKTUAL])
    / pred_df[PRED_COL_AKTUAL].clip(lower=1e-9) * 100
)
pred_df.to_csv(SAR_PRED_PATH, index=False)
summary_df.to_csv(SAR_SUMMARY_PATH, index=False)

print(f"  -> Prediksi LOOCV (3 model) : {SAR_PRED_PATH}")
print(f"  -> Ringkasan metrik         : {SAR_SUMMARY_PATH}")

print("\n  [SELESAI] SAR/SEM/OLS LOOCV pipeline selesai.")
