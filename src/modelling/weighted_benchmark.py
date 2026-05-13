"""
==============================================================================
 POVERTY MODELLING PIPELINE (WEIGHTED) - Estimasi Kemiskinan Kab. Sleman
==============================================================================
 Alur kerja:
   1. Load & inspect data
   2. EDA: deskripsi fitur
   3. StandardScaler
   4. Konstruksi bobot berbasis Cook's Distance invers (Cook 1977)
        - Observasi high-leverage / influential diberi bobot LEBIH KECIL
   5. LOOCV benchmark TANPA bobot (baseline)
   6. LOOCV benchmark DENGAN bobot Cook's Distance
   7. Perbandingan: baseline vs weighted per model
   8. Ringkasan & detail prediksi terbaik

 Referensi pembobotan:
   - Cook, R.D. (1977). Technometrics, 19(1), 15-18.
   - Goldberg et al. (1998). NIPS → Heteroskedastic GPR via alpha array.
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
from src.config import (
    MODEL_DATA_CENTROID_PATH, TARGET_COL, KECAMATAN_COL,
    LONG_COL, LAT_COL,
)


# ==============================================================================
# STEP 1 - LOAD DATA
# ==============================================================================
section("STEP 1 - LOAD DATA")

df = pd.read_csv(MODEL_DATA_CENTROID_PATH)

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

print(f"\n  Statistik deskriptif target ({TARGET_COL}):")
print(df[TARGET_COL].describe().to_string())

# ==============================================================================
# STEP 2 - EDA: DESKRIPSI FITUR
# ==============================================================================
section("STEP 2 - EDA - DESKRIPSI FITUR")

# Koordinat dan identifier TIDAK dipakai sebagai fitur model
exclude_cols = [KECAMATAN_COL, TARGET_COL, LONG_COL, LAT_COL]
feature_cols = [c for c in df.columns if c not in exclude_cols]
X_raw = df[feature_cols].copy()
y     = df[TARGET_COL].values

print(f"\n  Menggunakan semua fitur yang tersedia ({len(feature_cols)} fitur):")
for i, f in enumerate(feature_cols, 1):
    print(f"    {i:2}. {f}")

print(f"\n  Target: {TARGET_COL}")


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
# STEP 3.5 - KONSTRUKSI BOBOT: w_final berbasis Cook's Distance (Cook 1977)
# ==============================================================================
section("STEP 3.5 - KONSTRUKSI BOBOT COOK'S DISTANCE")

# ------------------------------------------------------------------------------
# Cook's Distance Inverse Weight (Cook, 1977)
#    Logika: observasi yang sangat berpengaruh (influential / high-leverage)
#    diberi bobot LEBIH KECIL agar tidak mendominasi fitting.
#    Bobot = 1 / (D_i + epsilon).
#    Dihitung dari model linear sederhana (OLS) sebagai diagnostik.
# ------------------------------------------------------------------------------
from numpy.linalg import lstsq, pinv

# Fit OLS sederhana (X_scaled → y) untuk hitung Cook's D
n, p   = X_scaled.shape
X_ols  = np.hstack([np.ones((n, 1)), X_scaled])   # tambah intercept
p_ols  = X_ols.shape[1]

beta_hat   = lstsq(X_ols, y, rcond=None)[0]
y_hat      = X_ols @ beta_hat
residuals  = y - y_hat
mse_ols    = np.sum(residuals**2) / (n - p_ols)

# Hat matrix H = X(X'X)^-1 X'
H          = X_ols @ pinv(X_ols.T @ X_ols) @ X_ols.T
h_ii       = np.diag(H)                            # leverage tiap observasi

# Cook's Distance: D_i = (e_i^2 / (p * MSE)) * (h_ii / (1-h_ii)^2)
cook_d     = (residuals**2 / (p_ols * mse_ols)) * (h_ii / (1 - h_ii)**2)

# Threshold umum untuk small sample: 4 / (n - p - 1)
threshold  = 4 / (n - p_ols - 1)

# Bobot invers: observasi berpengaruh tinggi → bobot kecil
eps = 1e-9
w_cook_raw = 1.0 / (cook_d + eps)
w_cook     = w_cook_raw / w_cook_raw.sum()         # normalisasi sum=1

print(f"\n  Threshold Cook's D (4/(n-p-1)) : {threshold:.4f}")
print(f"\n  {'Kecamatan':<16} {'Cook_D':>10} {'Influential?':>14} {'w_cook':>10}")
print(f"  {'-'*16} {'-'*10} {'-'*14} {'-'*10}")
for kec, cd, wc in zip(df[KECAMATAN_COL], cook_d, w_cook):
    flag = "  ⚠ YA" if cd > threshold else "  ok"
    print(f"  {kec:<16} {cd:>10.4f} {flag:<14} {wc:>10.6f}")

# ------------------------------------------------------------------------------
# w_final = w_cook (langsung digunakan, tanpa gabungan spasial)
# ------------------------------------------------------------------------------
w_final = w_cook

print(f"\n  Sum w_final = {w_final.sum():.6f}  (harus = 1.0)")
print(f"\n  Referensi:")
print(f"    * Cook (1977) Technometrics 19(1):15-18  → w_cook")


section("STEP 4 - LOOCV BENCHMARK - 6 Model + GPR Weighted (Baseline vs Weighted)")

print(f"\n  Strategi: Leave-One-Out CV (n={len(y)} fold)")
print("  Setiap fold: 1 kecamatan jadi test, sisanya train")
print("  Dijalankan DUA kali: tanpa bobot (baseline) & dengan w_final")
print("  PLUS satu run khusus: GPR Weighted via alpha array (heteroskedastic GPR)")
print("  Metrik: RMSE | MAE | MAPE")

loo = LeaveOneOut()

# -- Factory model (fresh instance tiap run agar tidak carry state) -----------
def get_models():
    return {
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

# Model yang tidak mendukung sample_weight
NO_WEIGHT_SUPPORT = {"Gaussian Process", "SVR (RBF)"}

# -- Fungsi LOOCV generik -----------------------------------------------------
def run_loocv(models_dict, X_sc, y_arr, df_ref, weights=None, label="BASE"):
    res = {}
    for name, model in models_dict.items():
        subsection(f"[{label}] {name}")

        use_weight = (weights is not None) and (name not in NO_WEIGHT_SUPPORT)
        if weights is not None and not use_weight:
            print(f"  [INFO] sample_weight tidak didukung → identik dengan baseline.")

        y_true_all, y_pred_all = [], []

        for fold, (train_idx, test_idx) in enumerate(loo.split(X_sc)):
            X_train, X_test = X_sc[train_idx], X_sc[test_idx]
            y_train, y_test = y_arr[train_idx], y_arr[test_idx]

            if use_weight:
                w_train = weights[train_idx]
                w_train = w_train / w_train.sum()  # re-normalisasi per fold
                model.fit(X_train, y_train, sample_weight=w_train)
            else:
                model.fit(X_train, y_train)

            pred = model.predict(X_test)[0]
            y_true_all.append(y_test[0])
            y_pred_all.append(pred)

            kec_name = df_ref[KECAMATAN_COL].iloc[test_idx[0]]
            print(f"  Fold {fold+1:>2} | {kec_name:<15} | "
                  f"aktual={y_test[0]:.2f}%  prediksi={pred:.2f}%  "
                  f"err={abs(y_test[0]-pred):.2f}")

        rmse_val = np.sqrt(mean_squared_error(y_true_all, y_pred_all))
        mae_val  = mean_absolute_error(y_true_all, y_pred_all)
        mape_val = mape(y_true_all, y_pred_all)

        res[name] = {"RMSE": rmse_val, "MAE": mae_val,
                     "MAPE": mape_val, "y_pred": y_pred_all}

        print(f"\n  > RMSE={rmse_val:.4f}  MAE={mae_val:.4f}  MAPE={mape_val:.2f}%")
    return res


# -- Run baseline (tanpa bobot) -----------------------------------------------
section("STEP 4A - BASELINE (tanpa bobot)")
results_base = run_loocv(get_models(), X_scaled, y, df, weights=None, label="BASE")

# -- Run weighted (w_final) ---------------------------------------------------
section("STEP 4B - WEIGHTED (w_cook — Cook 1977) — Ridge, Lasso, RF, XGB")
results_wgt  = run_loocv(get_models(), X_scaled, y, df, weights=w_final, label="WGT")

# -- Run GPR Weighted via alpha array -----------------------------------------
# Pendekatan: heteroskedastic GPR (Goldberg et al., NIPS 1998)
# alpha_i = sigma^2 / w_final_i → observasi bobot kecil dianggap noisy oleh GPR
# Ini berbeda dari sample_weight: bobot masuk lewat prior noise per titik,
# bukan lewat pengulangan observasi. Jalankan LOOCV manual agar alpha ikut di-slice.
section("STEP 4C - GPR WEIGHTED via alpha array (Heteroskedastic GPR)")

print(f"\n  Metode  : alpha_i = sigma_base^2 / w_final_i")
print(f"  Logika  : w_final kecil → alpha besar → GPR anggap titik itu noisy")
print(f"  Ref     : Goldberg, Williams & Bishop (1998) NIPS — Regression with")
print(f"            Input-Dependent Noise: A Gaussian Process Treatment")
print(f"  Kernel  : RBF + WhiteKernel (sama dengan GPR baseline)\n")

SIGMA_BASE  = 1.0    # noise dasar; akan diskalakan per observasi oleh w_final
eps_alpha   = 1e-9

y_true_gpr_w, y_pred_gpr_w = [], []

for fold, (train_idx, test_idx) in enumerate(loo.split(X_scaled)):
    X_train, X_test = X_scaled[train_idx], X_scaled[test_idx]
    y_train, y_test = y[train_idx],        y[test_idx]

    # Slice w_final ke training fold lalu hitung alpha per titik
    w_train      = w_final[train_idx]
    w_train_norm = w_train / w_train.sum()           # normalisasi fold

    # alpha_i = sigma_base^2 / w_i  →  titik berbobot rendah = lebih noisy
    alpha_arr = (SIGMA_BASE ** 2) / (w_train_norm + eps_alpha)

    gpr_w = GaussianProcessRegressor(
        kernel=RBF(length_scale=1.0) + WhiteKernel(noise_level=0.1),
        alpha=alpha_arr,              # <-- array, bukan scalar
        n_restarts_optimizer=10,
        random_state=42
    )
    gpr_w.fit(X_train, y_train)
    pred = gpr_w.predict(X_test)[0]

    y_true_gpr_w.append(y_test[0])
    y_pred_gpr_w.append(pred)

    kec_name = df[KECAMATAN_COL].iloc[test_idx[0]]
    print(f"  Fold {fold+1:>2} | {kec_name:<15} | "
          f"aktual={y_test[0]:.2f}%  prediksi={pred:.2f}%  "
          f"err={abs(y_test[0]-pred):.2f}  "
          f"alpha_max={alpha_arr.max():.3f}")

rmse_gw  = np.sqrt(mean_squared_error(y_true_gpr_w, y_pred_gpr_w))
mae_gw   = mean_absolute_error(y_true_gpr_w, y_pred_gpr_w)
mape_gw  = mape(y_true_gpr_w, y_pred_gpr_w)

result_gpr_weighted = {
    "RMSE": rmse_gw, "MAE": mae_gw,
    "MAPE": mape_gw, "y_pred": y_pred_gpr_w
}

print(f"\n  > RMSE={rmse_gw:.4f}  MAE={mae_gw:.4f}  MAPE={mape_gw:.2f}%")
print(f"  > GPR baseline MAPE = {results_base['Gaussian Process']['MAPE']:.2f}%")
delta_gpr = mape_gw - results_base["Gaussian Process"]["MAPE"]
verdict_gpr = "GPR Weighted lebih baik" if delta_gpr < -0.01 else (
              "GPR Baseline lebih baik" if delta_gpr >  0.01 else "setara")
print(f"  > Delta vs GPR baseline = {delta_gpr:+.2f}%  → {verdict_gpr}")


# ==============================================================================
# STEP 5 - PERBANDINGAN LENGKAP: BASELINE vs WEIGHTED vs GPR WEIGHTED
# ==============================================================================
section("STEP 5 - PERBANDINGAN LENGKAP")

print(f"\n  {'Model':<26} {'BASE MAPE':>10} {'WGT MAPE':>10} {'Delta':>8}  Kesimpulan")
print(f"  {'-'*26} {'-'*10} {'-'*10} {'-'*8}  {'-'*24}")

all_results_for_best = {}   # kumpulkan semua untuk cari overall best

for name in results_base:
    mb = results_base[name]["MAPE"]
    mw = results_wgt[name]["MAPE"]
    delta = mw - mb
    verdict = "WGT lebih baik" if delta < -0.01 else (
              "BASE lebih baik" if delta >  0.01 else "setara")
    print(f"  {name:<26} {mb:>9.2f}% {mw:>9.2f}% {delta:>+7.2f}%  {verdict}")
    all_results_for_best[f"{name} (BASE)"] = {"MAPE": mb, "y_pred": results_base[name]["y_pred"]}
    all_results_for_best[f"{name} (WGT)"]  = {"MAPE": mw, "y_pred": results_wgt[name]["y_pred"]}

# Tambahkan baris GPR Weighted secara khusus
mb_gpr = results_base["Gaussian Process"]["MAPE"]
delta_gpr_row = mape_gw - mb_gpr
verdict_gpr_row = "WGT lebih baik" if delta_gpr_row < -0.01 else (
                  "BASE lebih baik" if delta_gpr_row >  0.01 else "setara")
print(f"  {'GPR (Weighted-alpha)':<26} {mb_gpr:>9.2f}% {mape_gw:>9.2f}% "
      f"{delta_gpr_row:>+7.2f}%  {verdict_gpr_row}  ← heteroskedastic")
all_results_for_best["GPR (Weighted-alpha)"] = {"MAPE": mape_gw, "y_pred": y_pred_gpr_w}

# Cari overall best dari semua kombinasi
overall_best_name = min(all_results_for_best, key=lambda k: all_results_for_best[k]["MAPE"])
overall_best_mape = all_results_for_best[overall_best_name]["MAPE"]
overall_best_pred = all_results_for_best[overall_best_name]["y_pred"]

best_name_b = min(results_base, key=lambda k: results_base[k]["MAPE"])
best_name_w = min(results_wgt,  key=lambda k: results_wgt[k]["MAPE"])
best_mape_b = results_base[best_name_b]["MAPE"]
best_mape_w = results_wgt[best_name_w]["MAPE"]

print(f"""
  +--------------------------------------------------------------+
  |  BASELINE terbaik    : {best_name_b:<24} MAPE={best_mape_b:.2f}%  |
  |  WEIGHTED terbaik    : {best_name_w:<24} MAPE={best_mape_w:.2f}%  |
  |  GPR Weighted-alpha  : {'GPR (Weighted-alpha)':<24} MAPE={mape_gw:.2f}%  |
  +--------------------------------------------------------------+
  |  >> OVERALL TERBAIK  : {overall_best_name:<24} MAPE={overall_best_mape:.2f}%  |
  +--------------------------------------------------------------+
""")


# ==============================================================================
# STEP 6 - DETAIL PREDIKSI OVERALL TERBAIK
# ==============================================================================
section("STEP 6 - DETAIL PREDIKSI OVERALL TERBAIK")

print(f"\n  Model   : {overall_best_name}")
print(f"  MAPE    : {overall_best_mape:.2f}%")
if "GPR (Weighted-alpha)" in overall_best_name:
    print(f"  Metode  : Heteroskedastic GPR — alpha_i = sigma^2 / w_final_i")
    print(f"  Ref     : Goldberg, Williams & Bishop (1998) NIPS")
elif "WGT" in overall_best_name:
    print(f"  Bobot   : w_final = w_cook (Cook 1977) — inverse Cook's Distance")
else:
    print(f"  Bobot   : tidak ada (baseline)")

print(f"\n  {'Kecamatan':<16} {'Aktual (%)':>12} {'Prediksi (%)':>14} "
      f"{'Error (pp)':>12} {'APE (%)':>10} {'w_final':>10}")
print(f"  {'-'*16} {'-'*12} {'-'*14} {'-'*12} {'-'*10} {'-'*10}")

for kec, act, pred, wf in zip(df[KECAMATAN_COL], y, overall_best_pred, w_final):
    err = pred - act
    ape = abs(err / act) * 100 if act != 0 else 0
    print(f"  {kec:<16} {act:>12.2f} {pred:>14.2f} {err:>+12.2f} {ape:>9.2f}% {wf:>10.6f}")

print("  [SELESAI] Script pipeline weighted + GPR Weighted selesai dijalankan.")