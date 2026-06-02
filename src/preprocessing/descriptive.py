"""
descriptive.py
--------------
Statistik deskriptif untuk data final (kecamatan-level).
Gunakan `MODEL_DATA_PATH` dari config sebagai sumber data.
"""
import pandas as pd
import numpy as np

from src.config import MODEL_DATA_PATH, TARGET_COL

# =============================================================================
# 1. LOAD DATA
# =============================================================================
# Data final: sebagian besar kolom numerik pakai titik (.) sebagai desimal,
# tetapi TARGET_COL (persentase_penduduk_miskin) dikutip dan pakai koma (,).
# Strategi: baca dulu, lalu konversi kolom target secara manual.
df = pd.read_csv(MODEL_DATA_PATH)

# Konversi kolom target dari string berkoma desimal ke float
if TARGET_COL in df.columns and df[TARGET_COL].dtype == object:
    df[TARGET_COL] = (
        df[TARGET_COL]
        .astype(str)
        .str.replace(",", ".")
        .astype(float)
    )

# Pastikan kolom numerik lain dikonversi (force numeric, coerce error)
for col in df.columns:
    if col == "kecamatan":
        continue
    if df[col].dtype == object:
        df[col] = pd.to_numeric(df[col], errors="coerce")

print("=" * 70)
print("DATA FINAL — STATISTIK DESKRIPTIF")
print("=" * 70)
print(f"\nFile  : {MODEL_DATA_PATH}")
print(f"Shape : {df.shape[0]} baris × {df.shape[1]} kolom\n")

# =============================================================================
# 2. INFO DASAR
# =============================================================================
print("-" * 70)
print("2. TIPE DATA & NON-NULL COUNT")
print("-" * 70)
df.info()

# =============================================================================
# 3. MISSING VALUES
# =============================================================================
print("\n" + "-" * 70)
print("3. MISSING VALUES")
print("-" * 70)
missing = df.isnull().sum()
missing_pct = (df.isnull().mean() * 100).round(2)
missing_df = pd.DataFrame({"missing": missing, "pct": missing_pct})
missing_df = missing_df[missing_df["missing"] > 0].sort_values("missing", ascending=False)
if len(missing_df) == 0:
    print("✅ Tidak ada missing value.")
else:
    print(missing_df)

# =============================================================================
# 4. STATISTIK DESKRIPTIF (NUMERIK)
# =============================================================================
print("\n" + "-" * 70)
print("4. STATISTIK DESKRIPTIF — NUMERIK")
print("-" * 70)

num_cols = df.select_dtypes(include=[np.number]).columns.tolist()
desc = df[num_cols].describe(percentiles=[0.25, 0.5, 0.75]).T
desc["range"] = desc["max"] - desc["min"]
desc["iqr"] = desc["75%"] - desc["25%"]
desc["cv"] = (desc["std"] / desc["mean"]).round(4)  # coefficient of variation

print(desc.to_string(float_format="%.4f"))

# =============================================================================
# 5. SKEWNESS & KURTOSIS
# =============================================================================
print("\n" + "-" * 70)
print("5. SKEWNESS & KURTOSIS")
print("-" * 70)

skew_kurt = pd.DataFrame({
    "skewness": df[num_cols].skew().round(4),
    "kurtosis": df[num_cols].kurtosis().round(4),
})
print(skew_kurt.to_string())

# =============================================================================
# 6. KORELASI DENGAN TARGET
# =============================================================================
print("\n" + "-" * 70)
print(f"6. KORELASI DENGAN TARGET → '{TARGET_COL}'")
print("-" * 70)

if TARGET_COL in df.columns:
    # Pearson
    pearson_corr = df[num_cols].corr()[TARGET_COL].drop(TARGET_COL).sort_values(ascending=False)
    print("\n  [Pearson correlation]")
    print(pearson_corr.to_string(float_format="%.4f"))

    # Spearman
    spearman_corr = df[num_cols].corr(method="spearman")[TARGET_COL].drop(TARGET_COL).sort_values(ascending=False)
    print("\n  [Spearman correlation]")
    print(spearman_corr.to_string(float_format="%.4f"))
else:
    print(f"⚠ Kolom target '{TARGET_COL}' tidak ditemukan.")

# =============================================================================
# 7. KATEGORIKAL (jika ada selain kecamatan)
# =============================================================================
print("\n" + "-" * 70)
print("7. KOLOM KATEGORIKAL")
print("-" * 70)

cat_cols = df.select_dtypes(include=["object", "category"]).columns.tolist()
for col in cat_cols:
    print(f"\n  [{col}] — unique: {df[col].nunique()}")
    print(df[col].value_counts().to_string())

# =============================================================================
# 8. OUTLIER CHECK (IQR method)
# =============================================================================
print("\n" + "-" * 70)
print("8. OUTLIER CHECK (IQR method)")
print("-" * 70)

outlier_summary = []
for col in num_cols:
    Q1 = df[col].quantile(0.25)
    Q3 = df[col].quantile(0.75)
    IQR = Q3 - Q1
    lower = Q1 - 1.5 * IQR
    upper = Q3 + 1.5 * IQR
    n_outliers = ((df[col] < lower) | (df[col] > upper)).sum()
    if n_outliers > 0:
        outlier_summary.append((col, n_outliers, lower, upper))

if outlier_summary:
    for col, n, lo, hi in outlier_summary:
        print(f"  {col}: {n} outlier(s) → batas [{lo:.4f}, {hi:.4f}]")
else:
    print("✅ Tidak ada outlier.")

print("\n" + "=" * 70)
print("SELESAI")
print("=" * 70)
