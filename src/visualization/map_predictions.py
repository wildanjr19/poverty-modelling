"""
==============================================================================
 CHOROPLETH MAP - Estimasi Presentase Kemiskinan Kecamatan Sleman
==============================================================================
 Visualisasi hasil GPR terbaik ke peta 3-panel:
   Panel 1: Persentase Kemiskinan Aktual
   Panel 2: Persentase Kemiskinan Prediksi (GPR)
   Panel 3: Absolute Percentage Error (APE)
==============================================================================
"""

import os
import numpy as np
import pandas as pd
import geopandas as gpd
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
from matplotlib.patches import Patch
from matplotlib.colors import Normalize
from matplotlib.cm import ScalarMappable

# -- Paths -------------------------------------------------------------------
BASE_DIR = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
GEOJSON_PATH = os.path.join(BASE_DIR, "data", "geojson", "jumlah_jiwa_miskin_2024_1.json")
CSV_PATH = os.path.join(BASE_DIR, "outputs", "gpr_best_predictions.csv")
OUTPUT_PATH = os.path.join(BASE_DIR, "outputs", "map_kemiskinan_sleman.png")
OUTPUT_PATH_HIRES = os.path.join(BASE_DIR, "outputs", "map_kemiskinan_sleman_hires.png")

os.makedirs(os.path.dirname(OUTPUT_PATH), exist_ok=True)

# -- Load & Prepare GeoJSON --------------------------------------------------
print("[1/4] Loading GeoJSON...")
gdf = gpd.read_file(GEOJSON_PATH)  # CRS: EPSG:9489
print(f"      {len(gdf)} desa/kelurahan, CRS: {gdf.crs}")

# Dissolve to kecamatan level
gdf_kec = gdf.dissolve(by="wadmkc").reset_index()
print(f"      Dissolved -> {len(gdf_kec)} kecamatan")

# Reproject to WGS84 for contextily / lat-lon plotting
gdf_kec = gdf_kec.to_crs(epsg=4326)
print(f"      Reprojected to EPSG:4326")

# -- Load Predictions CSV ----------------------------------------------------
print("[2/4] Loading predictions...")
pred_df = pd.read_csv(CSV_PATH)
print(f"      {len(pred_df)} kecamatan loaded")
print(f"      Kolom: {list(pred_df.columns)}")
print(f"      MAPE: {pred_df['ape_pct'].mean():.2f}%  |  "
      f"RMSE: {np.sqrt(np.mean((pred_df['prediksi_pct'] - pred_df['aktual_pct'])**2)):.4f}")

# -- Merge GeoJSON + Predictions ----------------------------------------------
print("[3/4] Merging GeoJSON + predictions...")
gdf_map = gdf_kec.merge(pred_df, left_on="wadmkc", right_on="kecamatan", how="left")

# Verify no unmatched
unmatched = gdf_map[gdf_map["aktual_pct"].isna()]
if len(unmatched) > 0:
    print(f"      WARNING: {len(unmatched)} unmatched: {list(unmatched['wadmkc'].values)}")
else:
    print(f"      All {len(gdf_map)} kecamatan matched successfully.")

# -- Plot ---------------------------------------------------------------------
print("[4/4] Rendering 3-panel choropleth...")

# Color scheme
cmap_actual = plt.cm.YlOrRd          # kemiskinan: kuning -> merah
cmap_pred   = plt.cm.YlOrRd
cmap_ape    = plt.cm.RdYlGn_r        # error: merah (tinggi) -> hijau (rendah)

# Shared normalization for actual & pred (agar warna sebanding)
vmin = min(pred_df["aktual_pct"].min(), pred_df["prediksi_pct"].min())
vmax = max(pred_df["aktual_pct"].max(), pred_df["prediksi_pct"].max())

fig, axes = plt.subplots(1, 3, figsize=(24, 10), constrained_layout=True)

# -- Helper: plot choropleth with labels --------------------------------------
def plot_panel(ax, gdf, column, cmap, vmin, vmax, title, fmt="{x:.1f}%"):
    gdf.plot(
        column=column,
        cmap=cmap,
        edgecolor="black",
        linewidth=0.8,
        ax=ax,
        vmin=vmin,
        vmax=vmax,
        legend=False,
    )
    # Add kecamatan labels at centroid
    for _, row in gdf.iterrows():
        centroid = row.geometry.centroid
        val = row[column]
        if not pd.isna(val):
            ax.annotate(
                f"{row['wadmkc']}\n({val:.1f}%)",
                xy=(centroid.x, centroid.y),
                ha="center", va="center",
                fontsize=6.5, fontweight="bold",
                color="black",
                bbox=dict(boxstyle="round,pad=0.2", facecolor="white", alpha=0.75, edgecolor="gray", linewidth=0.5),
            )
    ax.set_title(title, fontsize=13, fontweight="bold", pad=12)
    ax.set_aspect("equal")
    ax.axis("off")

# Panel 1: Aktual
plot_panel(axes[0], gdf_map, "aktual_pct", cmap_actual, vmin, vmax,
           "a) Kemiskinan Aktual (%)")

# Panel 2: Prediksi
plot_panel(axes[1], gdf_map, "prediksi_pct", cmap_pred, vmin, vmax,
           "b) Kemiskinan Prediksi GPR (%)")

# Panel 3: APE — warna sendiri
norm_ape = Normalize(vmin=pred_df["ape_pct"].min(), vmax=pred_df["ape_pct"].max())
for _, row in gdf_map.iterrows():
    centroid = row.geometry.centroid
    ape_val = row["ape_pct"]
    if not pd.isna(ape_val):
        ax = axes[2]
        color = cmap_ape(norm_ape(ape_val))
        ax.annotate(
            f"{row['wadmkc']}\n({ape_val:.1f}%)",
            xy=(centroid.x, centroid.y),
            ha="center", va="center",
            fontsize=6.5, fontweight="bold",
            color="black",
            bbox=dict(boxstyle="round,pad=0.2", facecolor="white", alpha=0.75, edgecolor="gray", linewidth=0.5),
        )
gdf_map.plot(
    column="ape_pct",
    cmap=cmap_ape,
    edgecolor="black",
    linewidth=0.8,
    ax=axes[2],
    vmin=norm_ape.vmin,
    vmax=norm_ape.vmax,
    legend=False,
)
axes[2].set_title("c) Absolute Percentage Error (%)", fontsize=13, fontweight="bold", pad=12)
axes[2].set_aspect("equal")
axes[2].axis("off")

# -- Colorbar terpadu untuk panel 1 & 2 ---------------------------------------
cbar_ax = fig.add_axes([0.08, 0.06, 0.30, 0.025])
sm_actual = ScalarMappable(norm=Normalize(vmin=vmin, vmax=vmax), cmap=cmap_actual)
cbar = fig.colorbar(sm_actual, cax=cbar_ax, orientation="horizontal")
cbar.set_label("Persentase Kemiskinan (%)", fontsize=10)
cbar.ax.tick_params(labelsize=8)

# -- Colorbar untuk APE -------------------------------------------------------
cbar_ax2 = fig.add_axes([0.62, 0.06, 0.30, 0.025])
sm_ape = ScalarMappable(norm=norm_ape, cmap=cmap_ape)
cbar2 = fig.colorbar(sm_ape, cax=cbar_ax2, orientation="horizontal")
cbar2.set_label("Absolute Percentage Error (%)", fontsize=10)
cbar2.ax.tick_params(labelsize=8)

# -- Suptitle -----------------------------------------------------------------
fig.suptitle(
    "Estimasi Persentase Kemiskinan Kecamatan — Kabupaten Sleman\n"
    f"Model: Gaussian Process Regression  |  MAPE = {pred_df['ape_pct'].mean():.2f}%",
    fontsize=15, fontweight="bold", y=1.02,
)

# -- Simpan -------------------------------------------------------------------
fig.savefig(OUTPUT_PATH, dpi=150, bbox_inches="tight", facecolor="white")
print(f"      -> Gambar disimpan: {OUTPUT_PATH}")

fig.savefig(OUTPUT_PATH_HIRES, dpi=300, bbox_inches="tight", facecolor="white")
print(f"      -> Hi-res disimpan : {OUTPUT_PATH_HIRES}")

plt.close(fig)
print("\n  [SELESAI] Choropleth map selesai. Buka outputs/ untuk melihat hasil.")
