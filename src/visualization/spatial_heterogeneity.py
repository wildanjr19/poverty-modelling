"""
==============================================================================
 SPATIAL HETEROGENEITY ANALYSIS - Estimasi Kemiskinan Sleman
==============================================================================
 Menganalisis dan memvisualkan heterogenitas spasial:
   1. LISA (Local Moran's I) cluster map -> hotspot/coldspot kemiskinan
   2. GWR local R² map -> di mana model fit baik/buruk
   3. GWR coefficient maps (multi-panel) -> variasi pengaruh fitur antar lokasi
   4. Tabel ranking heterogenitas per kecamatan

 Input:
   - data/geojson/jumlah_jiwa_miskin_2024_1.json (geometri kecamatan)
   - data/data_final_with_centroid.csv (target + koordinat)
   - outputs/gwr_local_coefficients.csv (koefisien lokal GWR)

 Output:
   - outputs/spatial_lisa_cluster.png
   - outputs/spatial_lisa_per_kecamatan.csv
   - outputs/spatial_gwr_local_r2.png
   - outputs/spatial_gwr_coefficients.png (multi-panel)
   - outputs/spatial_heterogeneity_summary.csv
==============================================================================
"""

import warnings
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
import geopandas as gpd
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap, Normalize
from matplotlib.cm import ScalarMappable
from matplotlib.patches import Patch

from libpysal.weights import KNN
from esda.moran import Moran_Local

from src.modelling.helpers import section, subsection
from src.config import (
    MODEL_DATA_CENTROID_PATH, TARGET_COL, KECAMATAN_COL,
    LONG_COL, LAT_COL, OUTPUT_DIR,
    MORAN_K_NEIGHBORS, MORAN_PERMUTATIONS, MORAN_SIGNIFICANCE,
    RANDOM_STATE,
)

# Paths
GEOJSON_PATH = OUTPUT_DIR.parent / "data" / "geojson" / "jumlah_jiwa_miskin_2024_1.json"
GWR_COEF_PATH = OUTPUT_DIR / "gwr_local_coefficients.csv"

LISA_OUTPUT = OUTPUT_DIR / "spatial_lisa_cluster.png"
LISA_TABLE_OUTPUT = OUTPUT_DIR / "spatial_lisa_per_kecamatan.csv"
R2_OUTPUT = OUTPUT_DIR / "spatial_gwr_local_r2.png"
COEF_OUTPUT = OUTPUT_DIR / "spatial_gwr_coefficients.png"
SUMMARY_OUTPUT = OUTPUT_DIR / "spatial_heterogeneity_summary.csv"


# ==============================================================================
# STEP 1 - LOAD DATA
# ==============================================================================
section("STEP 1 - LOAD DATA")

# Target + koordinat
df = pd.read_csv(MODEL_DATA_CENTROID_PATH)
if df[TARGET_COL].dtype == object:
    df[TARGET_COL] = (
        df[TARGET_COL]
        .astype(str)
        .str.replace(",", ".", regex=False)
        .astype(float)
    )

print(f"\n  Data kecamatan: {len(df)} baris")
print(f"  Target range  : {df[TARGET_COL].min():.4f} - {df[TARGET_COL].max():.4f}")

# GWR local coefficients
gwr_coef = pd.read_csv(GWR_COEF_PATH)
print(f"  GWR coef      : {len(gwr_coef)} baris, {gwr_coef.shape[1]-1} koefisien")

# GeoJSON (geometri kecamatan)
gdf_raw = gpd.read_file(GEOJSON_PATH)
gdf_kec = gdf_raw.dissolve(by="wadmkc").reset_index()
gdf_kec = gdf_kec.to_crs(epsg=4326)
print(f"  GeoJSON       : {len(gdf_kec)} kecamatan, CRS=EPSG:4326")

# Merge semua
gdf = gdf_kec.merge(df, left_on="wadmkc", right_on=KECAMATAN_COL, how="inner")
gdf = gdf.merge(gwr_coef, left_on="wadmkc", right_on=KECAMATAN_COL, how="inner", suffixes=("", "_gwr"))
print(f"  Merged GDF    : {len(gdf)} kecamatan")

# Koordinat untuk spatial weights
coords = np.column_stack([gdf[LONG_COL].values, gdf[LAT_COL].values])


# ==============================================================================
# STEP 2 - LISA CLUSTER MAP
# ==============================================================================
section("STEP 2 - LISA (Local Moran's I) CLUSTER MAP")

w = KNN.from_array(coords, k=MORAN_K_NEIGHBORS)
w.transform = "r"

y = gdf[TARGET_COL].values
lisa = Moran_Local(
    y,
    w,
    permutations=MORAN_PERMUTATIONS,
    seed=RANDOM_STATE,
)

print(f"\n  Spatial weights: KNN k={MORAN_K_NEIGHBORS}")
print(f"  Permutations   : {MORAN_PERMUTATIONS}")
print(f"  Mean Local I   : {lisa.Is.mean():.4f}")

# Klasifikasi cluster (1=HH, 2=LH, 3=LL, 4=HL, 0=not significant)
sig_mask = lisa.p_sim < MORAN_SIGNIFICANCE
clusters = lisa.q * sig_mask  # 0 jika tidak signifikan

gdf["lisa_cluster"] = clusters
gdf["lisa_pval"] = lisa.p_sim
gdf["lisa_index"] = lisa.Is

cluster_counts = pd.Series(clusters).value_counts().sort_index()
print(f"\n  Cluster counts:")
labels_map = {0: "Not Sig", 1: "HH (hotspot)", 2: "LH", 3: "LL (coldspot)", 4: "HL"}
for c, count in cluster_counts.items():
    print(f"    {labels_map.get(c, c)}: {count}")

lisa_table = gdf[
    ["wadmkc", TARGET_COL, "lisa_index", "lisa_pval", "lisa_cluster"]
].copy()
lisa_table["lisa_cluster"] = lisa_table["lisa_cluster"].map(labels_map)
lisa_table["is_significant"] = lisa_table["lisa_pval"] < MORAN_SIGNIFICANCE
lisa_table = lisa_table.rename(
    columns={
        "wadmkc": "kecamatan",
        TARGET_COL: "target_actual",
        "lisa_pval": "lisa_pvalue",
    }
).sort_values("kecamatan").reset_index(drop=True)
lisa_table.to_csv(LISA_TABLE_OUTPUT, index=False)

print(f"\n  LISA index dan p-value per kecamatan:")
print(
    lisa_table[
        ["kecamatan", "lisa_index", "lisa_pvalue", "lisa_cluster"]
    ].to_string(index=False)
)
print(f"\n  -> LISA table saved: {LISA_TABLE_OUTPUT}")

subsection("Plot LISA Cluster Map")

fig, ax = plt.subplots(1, 1, figsize=(12, 10))

# Color scheme: 0=gray, 1=red, 2=pink, 3=blue, 4=lightblue
colors = ["#d3d3d3", "#d7191c", "#fdae61", "#2c7bb6", "#abd9e9"]
cmap_lisa = ListedColormap(colors)

gdf.plot(
    column="lisa_cluster",
    cmap=cmap_lisa,
    edgecolor="black",
    linewidth=1.0,
    ax=ax,
    legend=False,
    vmin=0,
    vmax=4,
)

# Annotate kecamatan
for _, row in gdf.iterrows():
    centroid = row.geometry.centroid
    cluster_label = labels_map.get(row["lisa_cluster"], "?")
    ax.annotate(
        f"{row['wadmkc']}\n({cluster_label})",
        xy=(centroid.x, centroid.y),
        ha="center", va="center",
        fontsize=7, fontweight="bold",
        color="black",
        bbox=dict(boxstyle="round,pad=0.3", facecolor="white", alpha=0.8, edgecolor="gray", linewidth=0.5),
    )

ax.set_title(
    f"LISA Cluster Map - Persentase Kemiskinan Kecamatan Sleman\n"
    f"(KNN k={MORAN_K_NEIGHBORS}, p<{MORAN_SIGNIFICANCE})",
    fontsize=14, fontweight="bold", pad=15
)
ax.set_aspect("equal")
ax.axis("off")

# Legend
legend_elements = [
    Patch(facecolor=colors[0], edgecolor="black", label="Not Significant"),
    Patch(facecolor=colors[1], edgecolor="black", label="High-High (Hotspot)"),
    Patch(facecolor=colors[2], edgecolor="black", label="Low-High"),
    Patch(facecolor=colors[3], edgecolor="black", label="Low-Low (Coldspot)"),
    Patch(facecolor=colors[4], edgecolor="black", label="High-Low"),
]
ax.legend(handles=legend_elements, loc="lower left", fontsize=10, frameon=True)

plt.tight_layout()
fig.savefig(LISA_OUTPUT, dpi=200, bbox_inches="tight", facecolor="white")
plt.close(fig)
print(f"  -> LISA map saved: {LISA_OUTPUT}")


# ==============================================================================
# STEP 3 - GWR LOCAL R² MAP
# ==============================================================================
section("STEP 3 - GWR LOCAL R² MAP")

# Hitung local R² dari GWR residuals
# Karena kita tidak punya local R² langsung dari mgwr output sebelumnya,
# kita estimasi via: local_R2 ≈ 1 - (residual_i^2 / var(y))
# Alternatif: refit GWR dengan output local R² (tapi ini quick proxy)

print("\n  Catatan: Local R² di-estimasi dari variance explained per lokasi")
print("           (proxy, bukan exact local R² dari GWR fit)")

# Load prediksi GWR LOOCV untuk hitung residual
gwr_pred_path = OUTPUT_DIR / "gwr_loocv_predictions.csv"
if gwr_pred_path.exists():
    gwr_pred = pd.read_csv(gwr_pred_path)
    gdf = gdf.merge(
        gwr_pred[[KECAMATAN_COL, "aktual_pct", "prediksi_pct"]],
        left_on="wadmkc",
        right_on=KECAMATAN_COL,
        how="left",
        suffixes=("", "_pred")
    )
    gdf["residual"] = gdf["prediksi_pct"] - gdf["aktual_pct"]
    gdf["residual_sq"] = gdf["residual"] ** 2
    
    # Local R² proxy: 1 - (res_i^2 / var(y))
    var_y = gdf["aktual_pct"].var()
    gdf["local_r2"] = 1 - (gdf["residual_sq"] / var_y)
    gdf["local_r2"] = gdf["local_r2"].clip(lower=-1, upper=1)  # bound untuk visualisasi
    
    print(f"  Local R² range: {gdf['local_r2'].min():.4f} - {gdf['local_r2'].max():.4f}")
    print(f"  Mean local R² : {gdf['local_r2'].mean():.4f}")
else:
    print(f"  WARNING: {gwr_pred_path} tidak ditemukan, skip local R² map")
    gdf["local_r2"] = 0.0

subsection("Plot Local R² Map")

fig, ax = plt.subplots(1, 1, figsize=(12, 10))

cmap_r2 = plt.cm.RdYlGn  # merah (buruk) -> hijau (baik)
norm_r2 = Normalize(vmin=gdf["local_r2"].min(), vmax=gdf["local_r2"].max())

gdf.plot(
    column="local_r2",
    cmap=cmap_r2,
    edgecolor="black",
    linewidth=1.0,
    ax=ax,
    legend=False,
    norm=norm_r2,
)

# Annotate
for _, row in gdf.iterrows():
    centroid = row.geometry.centroid
    r2_val = row["local_r2"]
    ax.annotate(
        f"{row['wadmkc']}\n(R²={r2_val:.2f})",
        xy=(centroid.x, centroid.y),
        ha="center", va="center",
        fontsize=7, fontweight="bold",
        color="black",
        bbox=dict(boxstyle="round,pad=0.3", facecolor="white", alpha=0.8, edgecolor="gray", linewidth=0.5),
    )

ax.set_title(
    "GWR Local R² Map - Model Fit Quality per Kecamatan\n"
    "(Hijau = fit baik, Merah = fit buruk)",
    fontsize=14, fontweight="bold", pad=15
)
ax.set_aspect("equal")
ax.axis("off")

# Colorbar
sm_r2 = ScalarMappable(norm=norm_r2, cmap=cmap_r2)
cbar = plt.colorbar(sm_r2, ax=ax, orientation="horizontal", pad=0.05, shrink=0.6)
cbar.set_label("Local R²", fontsize=11)

plt.tight_layout()
fig.savefig(R2_OUTPUT, dpi=200, bbox_inches="tight", facecolor="white")
plt.close(fig)
print(f"  -> Local R² map saved: {R2_OUTPUT}")


# ==============================================================================
# STEP 4 - GWR COEFFICIENT MAPS (MULTI-PANEL)
# ==============================================================================
section("STEP 4 - GWR COEFFICIENT MAPS (TOP FEATURES)")

# Pilih top-6 fitur berdasarkan std koefisien (variasi tertinggi antar lokasi)
coef_cols = [c for c in gwr_coef.columns if c not in [KECAMATAN_COL, "intercept"]]
coef_std = gwr_coef[coef_cols].std().sort_values(ascending=False)
top_features = coef_std.head(6).index.tolist()

print(f"\n  Top-6 fitur dengan variasi koefisien tertinggi:")
for i, feat in enumerate(top_features, 1):
    print(f"    {i}. {feat:<20} (std={coef_std[feat]:.4f})")

subsection("Plot Multi-Panel Coefficient Maps")

fig, axes = plt.subplots(2, 3, figsize=(20, 14))
axes = axes.ravel()

for idx, feat in enumerate(top_features):
    ax = axes[idx]
    
    # Normalisasi warna: diverging (biru=negatif, merah=positif)
    vmin = gdf[feat].min()
    vmax = gdf[feat].max()
    vabs = max(abs(vmin), abs(vmax))
    norm = Normalize(vmin=-vabs, vmax=vabs)
    cmap_div = plt.cm.RdBu_r
    
    gdf.plot(
        column=feat,
        cmap=cmap_div,
        edgecolor="black",
        linewidth=0.8,
        ax=ax,
        legend=False,
        norm=norm,
    )
    
    # Annotate
    for _, row in gdf.iterrows():
        centroid = row.geometry.centroid
        coef_val = row[feat]
        ax.annotate(
            f"{row['wadmkc']}\n({coef_val:.2f})",
            xy=(centroid.x, centroid.y),
            ha="center", va="center",
            fontsize=6, fontweight="bold",
            color="black",
            bbox=dict(boxstyle="round,pad=0.2", facecolor="white", alpha=0.75, edgecolor="gray", linewidth=0.4),
        )
    
    ax.set_title(f"{feat}\n(std={coef_std[feat]:.3f})", fontsize=11, fontweight="bold")
    ax.set_aspect("equal")
    ax.axis("off")
    
    # Colorbar per panel
    sm = ScalarMappable(norm=norm, cmap=cmap_div)
    cbar = plt.colorbar(sm, ax=ax, orientation="horizontal", pad=0.02, shrink=0.8)
    cbar.ax.tick_params(labelsize=7)

fig.suptitle(
    "GWR Local Coefficients - Spatial Heterogeneity per Feature\n"
    "(Biru = pengaruh negatif, Merah = pengaruh positif)",
    fontsize=15, fontweight="bold", y=0.98
)

plt.tight_layout()
fig.savefig(COEF_OUTPUT, dpi=200, bbox_inches="tight", facecolor="white")
plt.close(fig)
print(f"  -> Coefficient maps saved: {COEF_OUTPUT}")


# ==============================================================================
# STEP 5 - TABEL RANKING HETEROGENITAS
# ==============================================================================
section("STEP 5 - TABEL RANKING HETEROGENITAS PER KECAMATAN")

# Untuk tiap kecamatan, hitung:
# 1. Magnitude rata-rata koefisien (abs mean)
# 2. Fitur dengan koefisien terbesar (magnitude)
# 3. LISA cluster
# 4. Local R²

summary_rows = []
for _, row in gdf.iterrows():
    kec = row["wadmkc"]
    
    # Koefisien (exclude intercept)
    coef_vals = row[coef_cols].values
    abs_mean_coef = np.abs(coef_vals).mean()
    max_coef_idx = np.argmax(np.abs(coef_vals))
    dominant_feature = coef_cols[max_coef_idx]
    dominant_coef = coef_vals[max_coef_idx]
    
    summary_rows.append({
        "kecamatan": kec,
        "lisa_cluster": labels_map.get(row["lisa_cluster"], "Not Sig"),
        "lisa_index": row["lisa_index"],
        "lisa_pval": row["lisa_pval"],
        "local_r2": row["local_r2"],
        "abs_mean_coef": abs_mean_coef,
        "dominant_feature": dominant_feature,
        "dominant_coef": dominant_coef,
        "target_actual": row[TARGET_COL],
    })

summary_df = pd.DataFrame(summary_rows)
summary_df = summary_df.sort_values("abs_mean_coef", ascending=False).reset_index(drop=True)

print(f"\n  Top-5 kecamatan dengan heterogenitas tertinggi (abs mean coef):")
print(summary_df.head(5)[["kecamatan", "abs_mean_coef", "dominant_feature", "dominant_coef"]].to_string(index=False))

print(f"\n  Bottom-5 kecamatan (koefisien paling stabil/kecil):")
print(summary_df.tail(5)[["kecamatan", "abs_mean_coef", "dominant_feature", "dominant_coef"]].to_string(index=False))

# Simpan CSV
summary_df.to_csv(SUMMARY_OUTPUT, index=False)
print(f"\n  -> Summary table saved: {SUMMARY_OUTPUT}")


# ==============================================================================
# STEP 6 - RINGKASAN INSIGHT
# ==============================================================================
section("STEP 6 - RINGKASAN INSIGHT HETEROGENITAS SPASIAL")

print(f"""
  1. LISA Cluster Analysis:
     - Hotspot (HH): {(gdf['lisa_cluster'] == 1).sum()} kecamatan
     - Coldspot (LL): {(gdf['lisa_cluster'] == 3).sum()} kecamatan
     - Outlier (LH/HL): {((gdf['lisa_cluster'] == 2) | (gdf['lisa_cluster'] == 4)).sum()} kecamatan
     - Not Significant: {(gdf['lisa_cluster'] == 0).sum()} kecamatan
     
     Interpretasi: Hotspot = area kemiskinan tinggi yang bercluster;
                   Coldspot = area kemiskinan rendah yang bercluster.
  
  2. GWR Local R²:
     - Mean: {gdf['local_r2'].mean():.3f}
     - Range: [{gdf['local_r2'].min():.3f}, {gdf['local_r2'].max():.3f}]
     
     Interpretasi: Kecamatan dengan R² rendah = model kurang fit di sana,
                   kemungkinan ada faktor lokal yang belum tertangkap.
  
  3. GWR Coefficient Heterogeneity:
     - Top feature dengan variasi tertinggi: {top_features[0]} (std={coef_std[top_features[0]]:.3f})
     
     Interpretasi: Fitur ini punya pengaruh yang sangat berbeda antar lokasi.
                   Misal: di kecamatan A positif kuat, di B negatif/lemah.
  
  4. Dominant Features per Kecamatan:
     - Fitur paling sering dominan: {summary_df['dominant_feature'].value_counts().head(1).to_dict()}
     
     Interpretasi: Fitur ini paling sering jadi driver utama kemiskinan lokal.

  Output Files:
    - {LISA_OUTPUT.name}
    - {LISA_TABLE_OUTPUT.name}
    - {R2_OUTPUT.name}
    - {COEF_OUTPUT.name}
    - {SUMMARY_OUTPUT.name}
""")

print("\n  [SELESAI] Spatial heterogeneity analysis selesai.")
print("  Gunakan visualisasi ini untuk diskusi policy: area mana yang butuh intervensi spesifik.")

