"""
==============================================================================
 UJI MORAN'S I - Korelasi Spasial Persentase Penduduk Miskin
==============================================================================
 Sumber data : data/data_final_with_centroid.csv
 Variabel    : persentase_penduduk_miskin
 Koordinat   : long, lat (centroid kecamatan)
==============================================================================
"""

import warnings
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
from libpysal.weights import KNN
from esda.moran import Moran

# -----------------------------------------------------------------------------
# 1. LOAD DATA
# -----------------------------------------------------------------------------
df = pd.read_csv("data/data_final_with_centroid.csv")

# Fix locale-style decimal comma pada target
if df["persentase_penduduk_miskin"].dtype == object:
    df["persentase_penduduk_miskin"] = (
        df["persentase_penduduk_miskin"]
        .astype(str)
        .str.replace(",", ".", regex=False)
        .astype(float)
    )

# -----------------------------------------------------------------------------
# 2. EKSTRAK VARIABEL & KOORDINAT
# -----------------------------------------------------------------------------
y = df["persentase_penduduk_miskin"].values

# Koordinat centroid (longitude, latitude)
coords = df[["long", "lat"]].values

print("=" * 65)
print("  UJI MORAN'S I - Korelasi Spasial")
print("=" * 65)
print(f"\n  Jumlah observasi (kecamatan) : {len(y)}")
print(f"  Variabel uji                 : persentase_penduduk_miskin")
print(f"  Koordinat                    : long, lat")

# -----------------------------------------------------------------------------
# 3. BANGUN SPATIAL WEIGHTS MATRIX (K-Nearest Neighbors)
# -----------------------------------------------------------------------------
# Gunakan k=4 (umum untuk data jumlah sedang).
# Untuk 17 kecamatan, k tidak boleh >= n; k=4 dianggap aman.
k = 4 if len(y) > 5 else len(y) - 1
w = KNN.from_array(coords, k=k)

# Row-standardize weights (wij dijumlahkan per baris = 1)
w.transform = "r"

print(f"\n  Spatial weights matrix       : KNN (k={k})")
print(f"  Transformasi                 : Row-standardized")

# -----------------------------------------------------------------------------
# 4. HITUNG MORAN'S I
# -----------------------------------------------------------------------------
mi = Moran(y, w, permutations=999)

print("\n" + "-" * 65)
print("  HASIL UJI MORAN'S I")
print("-" * 65)
print(f"  Moran's I              : {mi.I:.4f}")
print(f"  Expected I (under H0)  : {mi.EI:.4f}")
print(f"  Variance               : {mi.VI_norm:.6f}")
print(f"  Z-score (normal)       : {mi.z_norm:.4f}")
print(f"  P-value (normal)       : {mi.p_norm:.4f}")
print("-" * 65)
print(f"  Moran's I (simulasi)   : {mi.I:.4f}")
print(f"  Z-score (simulasi)     : {mi.z_sim:.4f}")
print(f"  P-value (simulasi)     : {mi.p_sim:.4f}")
print("-" * 65)

# -----------------------------------------------------------------------------
# 5. INTERPRETASI
# -----------------------------------------------------------------------------
alpha = 0.05
print("\n  INTERPRETASI:")
if mi.p_norm < alpha:
    if mi.I > mi.EI:
        print("  -> Terdapat AUTOKORELASI SPASIAL POSITIF yang signifikan.")
        print("     Kecamatan dengan kemiskinan tinggi cenderung berdekatan")
        print("     dengan kecamatan lain yang juga memiliki kemiskinan tinggi.")
    else:
        print("  -> Terdapat AUTOKORELASI SPASIAL NEGATIF yang signifikan.")
        print("     Kecamatan dengan kemiskinan tinggi cenderung berdekatan")
        print("     dengan kecamatan lain yang memiliki kemiskinan rendah.")
else:
    print("  -> Tidak terdapat autokorelasi spasial yang signifikan.")
    print("     Distribusi kemiskinan antar kecamatan bersifat acak (random).")

print("\n" + "=" * 65)
print("  [SELESAI] Uji Moran's I selesai.")
print("=" * 65)
