"""
==============================================================================
 GPR KERNEL GRID SEARCH - BARE Kemiskinan Sleman
==============================================================================
 Eksperimen sistematis kombinasi kernel GPR dengan LOOCV.
 Alur:
   1. Load & prep data (sama seperti pipeline utama)
   2. Scale X dan y (target juga di-scale untuk GPR)
   3. Grid semua kombinasi kernel
   4. LOOCV tiap kernel -> RMSE, MAE, MAPE
   5. Ranking hasil + cetak prediksi model terbaik
==============================================================================
"""

import warnings
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import (
    RBF, Matern, WhiteKernel, DotProduct,
    RationalQuadratic, ConstantKernel as C
)
from sklearn.model_selection import LeaveOneOut
from sklearn.metrics import mean_squared_error, mean_absolute_error
import shap
import matplotlib.pyplot as plt

# -- Helpers ------------------------------------------------------------------
def section(title):
    print("\n" + "=" * 65)
    print(f"  {title}")
    print("=" * 65)

def subsection(title):
    print(f"\n-- {title} " + "-" * max(0, 55 - len(title)))

def mape(y_true, y_pred):
    y_true, y_pred = np.array(y_true), np.array(y_pred)
    mask = y_true != 0
    return np.mean(np.abs((y_true[mask] - y_pred[mask]) / y_true[mask])) * 100


# ==============================================================================
# STEP 1 - LOAD & PREP DATA
# ==============================================================================
section("STEP 1 - LOAD & PREP DATA")

df = pd.read_csv("data/data_final.csv")

# Fix locale-style decimal comma in target column
if df["persentase_penduduk_miskin"].dtype == object:
    df["persentase_penduduk_miskin"] = (
        df["persentase_penduduk_miskin"]
        .astype(str)
        .str.replace(",", ".", regex=False)
        .astype(float)
    )

print(f"\n  Baris: {len(df)}  |  Kolom: {df.shape[1]}")

# Fitur terpilih dari pipeline utama (top-3 korelasi)
# selected_features = ["ntl_mean", "population_mean", "dist_hospital"]
# X_raw = df[selected_features].values

# Gunakan semua fitur (kecuali target dan kolom non-fitur)
non_feature_cols = ["kecamatan", "persentase_penduduk_miskin"]
feature_cols = [c for c in df.columns if c not in non_feature_cols]
X_raw = df[feature_cols].values
y_raw = df["persentase_penduduk_miskin"].values

print(f"  Fitur : {feature_cols}  (total: {len(feature_cols)})")
print(f"  Target: persentase_penduduk_miskin  ->  min={y_raw.min():.2f}%  max={y_raw.max():.2f}%  mean={y_raw.mean():.2f}%")


# ==============================================================================
# STEP 2 - SCALING X DAN Y
# ==============================================================================
section("STEP 2 - SCALING - X dan y di-scale")

scaler_X = StandardScaler()
scaler_y = StandardScaler()

X_scaled = scaler_X.fit_transform(X_raw)
y_scaled = scaler_y.fit_transform(y_raw.reshape(-1, 1)).ravel()

print(f"\n  X : mean~{X_scaled.mean(axis=0).round(4)}  std~{X_scaled.std(axis=0).round(4)}")
print(f"  y : mean~{y_scaled.mean():.4f}  std~{y_scaled.std():.4f}")
print("  -> Prediksi akan di-inverse transform kembali ke skala % asli")


# ==============================================================================
# STEP 3 - DEFINISI GRID KERNEL
# ==============================================================================
section("STEP 3 - DEFINISI GRID KERNEL")

# Setiap entry: (nama_label, kernel_object, alpha)
# alpha = noise regularizer GPR (makin besar = makin toleran noise)

kernel_grid = [
    # -- RBF variants ------------------------------------------------------
    (
        "RBF + White (baseline)",
        C(1.0) * RBF(length_scale=1.0) + WhiteKernel(noise_level=0.1),
        1e-10
    ),
    (
        "RBF + White + alpha=0.5",
        C(1.0) * RBF(length_scale=1.0) + WhiteKernel(noise_level=0.1),
        0.5
    ),
    (
        "RBF + White + alpha=1.0",
        C(1.0) * RBF(length_scale=1.0) + WhiteKernel(noise_level=0.1),
        1.0
    ),

    # -- Matern variants ---------------------------------------------------
    (
        "Matern(nu=0.5) + White",
        C(1.0) * Matern(length_scale=1.0, nu=0.5) + WhiteKernel(noise_level=0.1),
        1e-10
    ),
    (
        "Matern(nu=1.5) + White",
        C(1.0) * Matern(length_scale=1.0, nu=1.5) + WhiteKernel(noise_level=0.1),
        1e-10
    ),
    (
        "Matern(nu=2.5) + White",
        C(1.0) * Matern(length_scale=1.0, nu=2.5) + WhiteKernel(noise_level=0.1),
        1e-10
    ),
    (
        "Matern(nu=2.5) + White + alpha=0.5",
        C(1.0) * Matern(length_scale=1.0, nu=2.5) + WhiteKernel(noise_level=0.1),
        0.5
    ),
    (
        "Matern(nu=2.5) + White + alpha=1.0",
        C(1.0) * Matern(length_scale=1.0, nu=2.5) + WhiteKernel(noise_level=0.1),
        1.0
    ),

    # -- RationalQuadratic -------------------------------------------------
    (
        "RatQuad + White",
        C(1.0) * RationalQuadratic(length_scale=1.0, alpha=1.0)
        + WhiteKernel(noise_level=0.1),
        1e-10
    ),
    (
        "RatQuad + White + alpha=0.5",
        C(1.0) * RationalQuadratic(length_scale=1.0, alpha=1.0)
        + WhiteKernel(noise_level=0.1),
        0.5
    ),

    # -- DotProduct (linear trend) -----------------------------------------
    (
        "DotProduct + White",
        DotProduct(sigma_0=1.0) + WhiteKernel(noise_level=0.1),
        1e-10
    ),
    (
        "DotProduct + White + alpha=0.5",
        DotProduct(sigma_0=1.0) + WhiteKernel(noise_level=0.1),
        0.5
    ),

    # -- Kombinasi (RBF + Matern) ------------------------------------------
    (
        "RBF + Matern(1.5) + White",
        C(1.0) * RBF(length_scale=1.0)
        + C(1.0) * Matern(length_scale=1.0, nu=1.5)
        + WhiteKernel(noise_level=0.1),
        1e-10
    ),
    (
        "RBF + Matern(2.5) + White",
        C(1.0) * RBF(length_scale=1.0)
        + C(1.0) * Matern(length_scale=1.0, nu=2.5)
        + WhiteKernel(noise_level=0.1),
        1e-10
    ),

    # -- Matern + DotProduct (trend + lokal) -------------------------------
    (
        "Matern(2.5) + DotProduct + White",
        C(1.0) * Matern(length_scale=1.0, nu=2.5)
        + DotProduct(sigma_0=1.0)
        + WhiteKernel(noise_level=0.1),
        1e-10
    ),
    (
        "Matern(2.5) + DotProduct + White + alpha=0.5",
        C(1.0) * Matern(length_scale=1.0, nu=2.5)
        + DotProduct(sigma_0=1.0)
        + WhiteKernel(noise_level=0.1),
        0.5
    ),
]

print(f"\n  Total kombinasi kernel yang akan diuji: {len(kernel_grid)}")
for i, (name, _, alpha) in enumerate(kernel_grid, 1):
    print(f"  {i:>2}. {name}  [alpha={alpha}]")


# ==============================================================================
# STEP 4 - LOOCV TIAP KERNEL
# ==============================================================================
section("STEP 4 - LOOCV - SEMUA KOMBINASI KERNEL")

loo     = LeaveOneOut()
results = []

for idx, (name, kernel, alpha_val) in enumerate(kernel_grid, 1):
    print(f"\n  [{idx:>2}/{len(kernel_grid)}] {name}")

    gpr = GaussianProcessRegressor(
        kernel=kernel,
        alpha=alpha_val,
        n_restarts_optimizer=10,   # naikkan ke 30-50 untuk hasil lebih stabil
        normalize_y=False,         # kita sudah scale y manual
        random_state=42
    )

    y_true_all, y_pred_all = [], []

    for train_idx, test_idx in loo.split(X_scaled):
        X_train, X_test = X_scaled[train_idx], X_scaled[test_idx]
        y_train, y_test = y_scaled[train_idx], y_scaled[test_idx]

        gpr.fit(X_train, y_train)
        pred_scaled = gpr.predict(X_test)[0]

        # Inverse transform kembali ke skala % asli
        pred_orig = scaler_y.inverse_transform([[pred_scaled]])[0][0]
        true_orig = scaler_y.inverse_transform([[y_test[0]]])[0][0]

        y_true_all.append(true_orig)
        y_pred_all.append(pred_orig)

    rmse_val = np.sqrt(mean_squared_error(y_true_all, y_pred_all))
    mae_val  = mean_absolute_error(y_true_all, y_pred_all)
    mape_val = mape(y_true_all, y_pred_all)

    results.append({
        "name"  : name,
        "RMSE"  : rmse_val,
        "MAE"   : mae_val,
        "MAPE"  : mape_val,
        "y_pred": y_pred_all,
    })

    print(f"       RMSE={rmse_val:.4f}  MAE={mae_val:.4f}  MAPE={mape_val:.2f}%")


# ==============================================================================
# STEP 5 - RANKING & HASIL TERBAIK
# ==============================================================================
section("STEP 5 - RANKING HASIL - DIURUTKAN MAPE")

results_sorted = sorted(results, key=lambda x: x["MAPE"])

print(f"\n  {'#':<4} {'Kernel':<45} {'RMSE':>7} {'MAE':>7} {'MAPE':>9}")
print(f"  {'-'*4} {'-'*45} {'-'*7} {'-'*7} {'-'*9}")
for i, r in enumerate(results_sorted, 1):
    marker = "  <- TERBAIK" if i == 1 else ("  <- top-3" if i <= 3 else "")
    print(f"  {i:<4} {r['name']:<45} {r['RMSE']:>7.4f} {r['MAE']:>7.4f} {r['MAPE']:>8.2f}%{marker}")

best = results_sorted[0]
print(f"""
  +---------------------------------------------------------+
  |  KERNEL TERBAIK : {best['name']:<38} |
  |  RMSE           : {best['RMSE']:.4f}                                  |
  |  MAE            : {best['MAE']:.4f}                                  |
  |  MAPE (LOOCV)   : {best['MAPE']:.2f}%                                 |
  +---------------------------------------------------------+
""")

# -- Detail prediksi kernel terbaik -------------------------------------------
subsection(f"Detail Prediksi - {best['name']}")

print(f"\n  {'Kecamatan':<16} {'Aktual (%)':>12} {'Prediksi (%)':>14} {'Error (pp)':>12} {'APE (%)':>10}")
print(f"  {'-'*16} {'-'*12} {'-'*14} {'-'*12} {'-'*10}")

for kec, act, pred in zip(df["kecamatan"], y_raw, best["y_pred"]):
    err  = pred - act
    ape  = abs(err / act) * 100 if act != 0 else 0
    print(f"  {kec:<16} {act:>12.2f} {pred:>14.2f} {err:>+12.2f} {ape:>9.2f}%")

print(f"""
------------------------------------------------------------------
  INTERPRETASI:
  * APE (Absolute Percentage Error) per kecamatan ditampilkan
    untuk identifikasi mana yang paling sulit diprediksi.
  * Kecamatan APE > 30%: pertimbangkan cek anomali data atau
    tambah fitur spesifik wilayah tersebut.
  * Jika MAPE masih > 15%, batas bawahnya kemungkinan memang
    dari keterbatasan n=17 - bukan masalah model.
------------------------------------------------------------------
""")

# ==============================================================================
# STEP 6 - SHAP ANALYSIS (Best Kernel)
# ==============================================================================
section("STEP 6 - SHAP ANALYSIS - Best GPR Model")

# Retrain best model on full scaled data
best_kernel_obj = None
best_alpha_val  = None
for name, kernel_obj, alpha_val in kernel_grid:
    if name == best['name']:
        best_kernel_obj = kernel_obj
        best_alpha_val  = alpha_val
        break

gpr_best_full = GaussianProcessRegressor(
    kernel=best_kernel_obj,
    alpha=best_alpha_val,
    n_restarts_optimizer=10,
    normalize_y=False,
    random_state=42
)
gpr_best_full.fit(X_scaled, y_scaled)

# Wrapper: kembalikan prediksi ke skala asli (%)
def gpr_predict_original(X):
    pred_scaled = gpr_best_full.predict(X)
    return scaler_y.inverse_transform(pred_scaled.reshape(-1, 1)).ravel()

subsection("SHAP KernelExplainer")

# n=17 sangat kecil -> KernelExplainer cepat
explainer = shap.KernelExplainer(gpr_predict_original, X_scaled)
shap_values = explainer.shap_values(X_scaled)

explanation = shap.Explanation(
    values=shap_values,
    base_values=np.full(X_scaled.shape[0], explainer.expected_value),
    data=X_scaled,
    feature_names=feature_cols
)

# --- Beeswarm ---
shap.plots.beeswarm(explanation, show=False)
plt.tight_layout()
out_beeswarm = "outputs/shap_beeswarm_best_gpr.png"
plt.savefig(out_beeswarm, dpi=150, bbox_inches="tight")
plt.close()
print(f"  -> Beeswarm plot disimpan: {out_beeswarm}")

# --- Bar ---
shap.plots.bar(explanation, show=False)
plt.tight_layout()
out_bar = "outputs/shap_bar_best_gpr.png"
plt.savefig(out_bar, dpi=150, bbox_inches="tight")
plt.close()
print(f"  -> Bar plot disimpan     : {out_bar}")

print(f"\n  Base value (rata-rata prediksi): {explainer.expected_value:.4f}%")
print("  Catatan: SHAP values merepresentasikan kontribusi tiap fitur")
print("           terhadap prediksi persentase kemiskinan (%) pada skala asli.")

print("\n  [SELESAI] GPR kernel grid search + SHAP selesai.")
