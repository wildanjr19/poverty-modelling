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
import geopandas as gpd
import matplotlib.pyplot as plt
from matplotlib.patches import Patch
from libpysal.weights import KNN
from libpysal.weights.spatial_lag import lag_spatial
from esda.moran import Moran

from src.config import (
    ADMIN_DESA_GEOJSON,
    ADMIN_KEC_COL,
    KECAMATAN_COL,
    LAT_COL,
    LONG_COL,
    MODEL_DATA_CENTROID_PATH,
    MORAN_K_NEIGHBORS,
    MORAN_PERMUTATIONS,
    OUTPUT_DIR,
    RANDOM_STATE,
    TARGET_COL,
)

MORAN_SCATTER_OUTPUT = OUTPUT_DIR / "moran_quadrant_scatter.png"
MORAN_MAP_OUTPUT = OUTPUT_DIR / "moran_quadrant_map.png"
MORAN_TABLE_OUTPUT = OUTPUT_DIR / "moran_quadrant_per_kecamatan.csv"

QUADRANT_LABELS = {
    "HH": "High-High",
    "LH": "Low-High",
    "LL": "Low-Low",
    "HL": "High-Low",
}
QUADRANT_COLORS = {
    "HH": "#d7191c",
    "LH": "#fdae61",
    "LL": "#2c7bb6",
    "HL": "#abd9e9",
}
QUADRANT_ORDER = ["HH", "LH", "LL", "HL"]


def classify_moran_quadrant(z_value, lag_value):
    """Classify Moran scatterplot quadrants from standardized value and lag."""
    if z_value >= 0 and lag_value >= 0:
        return "HH"
    if z_value < 0 and lag_value >= 0:
        return "LH"
    if z_value < 0 and lag_value < 0:
        return "LL"
    return "HL"


def save_moran_scatter(df_plot, moran_i):
    fig, ax = plt.subplots(figsize=(11, 8))
    label_offsets = {
        "Berbah": (10, -9),
        "Kalasan": (10, 8),
    }

    for quadrant in QUADRANT_ORDER:
        part = df_plot[df_plot["moran_quadrant"] == quadrant]
        ax.scatter(
            part["moran_z"],
            part["moran_lag_z"],
            s=80,
            color=QUADRANT_COLORS[quadrant],
            edgecolor="black",
            linewidth=0.8,
            label=f"{quadrant}: {QUADRANT_LABELS[quadrant]}",
            alpha=0.9,
        )

    for _, row in df_plot.iterrows():
        offset = label_offsets.get(row[KECAMATAN_COL], (5, 5))
        ax.annotate(
            row[KECAMATAN_COL],
            (row["moran_z"], row["moran_lag_z"]),
            xytext=offset,
            textcoords="offset points",
            fontsize=8,
        )

    x_min = df_plot["moran_z"].min() - 0.35
    x_max = df_plot["moran_z"].max() + 0.35
    x_line = np.linspace(x_min, x_max, 100)
    ax.plot(
        x_line,
        moran_i * x_line,
        color="#222222",
        linestyle="--",
        linewidth=1.2,
        label=f"Fit Moran's I = {moran_i:.3f}",
    )

    ax.axhline(0, color="black", linewidth=1.0)
    ax.axvline(0, color="black", linewidth=1.0)
    ax.grid(True, linestyle=":", linewidth=0.6, alpha=0.6)
    ax.set_xlim(x_min, x_max)
    ax.set_xlabel("Persentase penduduk miskin terstandar (z)")
    ax.set_ylabel("Spatial lag dari z")
    ax.set_title(
        "Moran Scatterplot - Kuadran Kecamatan\n"
        "Persentase Penduduk Miskin Kabupaten Sleman",
        fontsize=13,
        fontweight="bold",
        pad=12,
    )
    ax.legend(loc="best", fontsize=9, frameon=True)

    plt.tight_layout()
    fig.savefig(MORAN_SCATTER_OUTPUT, dpi=200, bbox_inches="tight", facecolor="white")
    plt.close(fig)


def save_moran_quadrant_map(df_plot):
    gdf_raw = gpd.read_file(ADMIN_DESA_GEOJSON)
    gdf_kec = gdf_raw.dissolve(by=ADMIN_KEC_COL).reset_index().to_crs(epsg=4326)

    gdf_map = gdf_kec.merge(
        df_plot[
            [
                KECAMATAN_COL,
                TARGET_COL,
                "moran_z",
                "moran_lag_z",
                "moran_quadrant",
                "moran_quadrant_label",
            ]
        ],
        left_on=ADMIN_KEC_COL,
        right_on=KECAMATAN_COL,
        how="left",
    )

    unmatched = gdf_map[gdf_map["moran_quadrant"].isna()][ADMIN_KEC_COL].tolist()
    if unmatched:
        print(f"\n  WARNING: Kecamatan tidak match di GeoJSON: {unmatched}")

    fig, ax = plt.subplots(figsize=(12, 10))

    for quadrant in QUADRANT_ORDER:
        part = gdf_map[gdf_map["moran_quadrant"] == quadrant]
        if len(part) == 0:
            continue
        part.plot(
            ax=ax,
            color=QUADRANT_COLORS[quadrant],
            edgecolor="black",
            linewidth=0.9,
        )

    missing = gdf_map[gdf_map["moran_quadrant"].isna()]
    if len(missing) > 0:
        missing.plot(ax=ax, color="#d9d9d9", edgecolor="black", linewidth=0.9)

    for _, row in gdf_map.dropna(subset=["moran_quadrant"]).iterrows():
        point = row.geometry.representative_point()
        ax.annotate(
            f"{row[ADMIN_KEC_COL]}\n{row['moran_quadrant']}",
            xy=(point.x, point.y),
            ha="center",
            va="center",
            fontsize=7,
            fontweight="bold",
            color="black",
            bbox=dict(
                boxstyle="round,pad=0.25",
                facecolor="white",
                alpha=0.78,
                edgecolor="gray",
                linewidth=0.5,
            ),
        )

    legend_elements = [
        Patch(
            facecolor=QUADRANT_COLORS[q],
            edgecolor="black",
            label=f"{q}: {QUADRANT_LABELS[q]}",
        )
        for q in QUADRANT_ORDER
    ]
    if len(missing) > 0:
        legend_elements.append(
            Patch(facecolor="#d9d9d9", edgecolor="black", label="Tidak match")
        )

    ax.legend(handles=legend_elements, loc="lower left", fontsize=10, frameon=True)
    ax.set_title(
        "Peta Kuadran Moran - Persentase Penduduk Miskin\n"
        "Kabupaten Sleman per Kecamatan",
        fontsize=14,
        fontweight="bold",
        pad=14,
    )
    ax.set_aspect("equal")
    ax.axis("off")

    plt.tight_layout()
    fig.savefig(MORAN_MAP_OUTPUT, dpi=200, bbox_inches="tight", facecolor="white")
    plt.close(fig)

# -----------------------------------------------------------------------------
# 1. LOAD DATA
# -----------------------------------------------------------------------------
df = pd.read_csv(MODEL_DATA_CENTROID_PATH)

# Fix locale-style decimal comma pada target
if df[TARGET_COL].dtype == object:
    df[TARGET_COL] = (
        df[TARGET_COL]
        .astype(str)
        .str.replace(",", ".", regex=False)
        .astype(float)
    )

# -----------------------------------------------------------------------------
# 2. EKSTRAK VARIABEL & KOORDINAT
# -----------------------------------------------------------------------------
y = df[TARGET_COL].values

# Koordinat centroid (longitude, latitude)
coords = df[[LONG_COL, LAT_COL]].values

print("=" * 65)
print("  UJI MORAN'S I - Korelasi Spasial")
print("=" * 65)
print(f"\n  Jumlah observasi (kecamatan) : {len(y)}")
print(f"  Variabel uji                 : {TARGET_COL}")
print(f"  Koordinat                    : long, lat")

# -----------------------------------------------------------------------------
# 3. BANGUN SPATIAL WEIGHTS MATRIX (K-Nearest Neighbors)
# -----------------------------------------------------------------------------
k = min(MORAN_K_NEIGHBORS, len(y) - 1)
w = KNN.from_array(coords, k=k)

# Row-standardize weights (wij dijumlahkan per baris = 1)
w.transform = "r"

print(f"\n  Spatial weights matrix       : KNN (k={k})")
print(f"  Transformasi                 : Row-standardized")

# -----------------------------------------------------------------------------
# 4. HITUNG MORAN'S I
# -----------------------------------------------------------------------------
np.random.seed(RANDOM_STATE)
mi = Moran(y, w, permutations=MORAN_PERMUTATIONS)

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
# 5. KUADRAN MORAN SCATTERPLOT
# -----------------------------------------------------------------------------
z = (y - y.mean()) / y.std(ddof=0)
lag_z = lag_spatial(w, z)

df["moran_z"] = z
df["moran_lag_z"] = lag_z
df["moran_quadrant"] = [
    classify_moran_quadrant(z_val, lag_val)
    for z_val, lag_val in zip(z, lag_z)
]
df["moran_quadrant_label"] = df["moran_quadrant"].map(QUADRANT_LABELS)

quadrant_table = df[
    [
        KECAMATAN_COL,
        TARGET_COL,
        "moran_z",
        "moran_lag_z",
        "moran_quadrant",
        "moran_quadrant_label",
    ]
].sort_values(["moran_quadrant", KECAMATAN_COL]).reset_index(drop=True)
quadrant_table.to_csv(MORAN_TABLE_OUTPUT, index=False)

print("\n" + "-" * 65)
print("  KUADRAN MORAN PER KECAMATAN")
print("-" * 65)
print(
    quadrant_table[
        [KECAMATAN_COL, "moran_z", "moran_lag_z", "moran_quadrant_label"]
    ].to_string(index=False)
)

save_moran_scatter(df, mi.I)
save_moran_quadrant_map(df)

print("\n  OUTPUT VISUALISASI:")
print(f"  -> Scatter kuadran : {MORAN_SCATTER_OUTPUT}")
print(f"  -> Peta kuadran    : {MORAN_MAP_OUTPUT}")
print(f"  -> Tabel kuadran   : {MORAN_TABLE_OUTPUT}")

# -----------------------------------------------------------------------------
# 6. INTERPRETASI
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
