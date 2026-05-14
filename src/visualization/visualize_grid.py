"""
==============================================================================
MAP - Kabupaten DIY (Grid 2km)
==============================================================================
"""

import os
import geopandas as gpd
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches

# Path konfigurasi
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_DIR = os.path.join(BASE_DIR, "data")
OUTPUT_DIR = os.path.join(BASE_DIR, "outputs")
os.makedirs(OUTPUT_DIR, exist_ok=True)

# Daftar file GeoJSON
files = {
    "Bantul": "bantul_grid_2km.geojson",
    "Gunungkidul": "gunungkidul_grid_2km.geojson",
    "Kota Yogyakarta": "kotayogyakarta_grid_2km.geojson",
    "Kulon Progo": "kulonprogo_grid_2km.geojson",
    "Sleman": "sleman_grid_2km.geojson",
}

# Warna untuk tiap kabupaten
colors = {
    "Bantul": "#e41a1c",
    "Gunungkidul": "#377eb8",
    "Kota Yogyakarta": "#4daf4a",
    "Kulon Progo": "#984ea3",
    "Sleman": "#ff7f00",
}

fig, ax = plt.subplots(figsize=(14, 14))

legend_patches = []

for kab_name, filename in files.items():
    filepath = os.path.join(DATA_DIR, filename)
    if not os.path.exists(filepath):
        print(f"[SKIP] File tidak ditemukan: {filepath}")
        continue

    gdf = gpd.read_file(filepath)

    # Pastikan CRS konsisten (WGS84)
    if gdf.crs is None:
        gdf.set_crs(epsg=4326, inplace=True)
    else:
        gdf = gdf.to_crs(epsg=4326)

    # Plot polygon dengan warna transparan dan border
    gdf.plot(
        ax=ax,
        facecolor=colors[kab_name],
        edgecolor="black",
        alpha=0.3,
        linewidth=0.5,
    )

    legend_patches.append(
        mpatches.Patch(facecolor=colors[kab_name], edgecolor="black", label=kab_name, alpha=0.6)
    )

    print(f"[OK] {kab_name}: {len(gdf)} grid")

ax.set_title("Visualisasi Grid 2km - DIY", fontsize=16, fontweight="bold")
ax.set_xlabel("Longitude", fontsize=12)
ax.set_ylabel("Latitude", fontsize=12)
ax.legend(handles=legend_patches, loc="upper left", fontsize=10)
ax.set_aspect("equal")
ax.grid(True, linestyle="--", alpha=0.5)

# Simpan gambar
output_path = os.path.join(OUTPUT_DIR, "visualisasi_grid_diy.png")
plt.tight_layout()
plt.savefig(output_path, dpi=300, bbox_inches="tight")
print(f"\nGambar disimpan di: {output_path}")

plt.show()
