"""
==============================================================================
 EXTRACT KECAMATAN BOUNDARIES — Kabupaten Sleman, DIY
==============================================================================
 Menghasilkan GeoJSON batas-batas 17 kecamatan di Kabupaten Sleman.

 CATATAN GADM:
   - gadm41_IDN_2.json = Level 2 (Kabupaten) → hanya outline Sleman saja
   - GADM Level 3 (Kecamatan) = gadm41_IDN_3.json → TIDAK tersedia di repo
   - Solusi: dissolve desa/kelurahan dari jumlah_jiwa_miskin_2024_1.json
     berdasarkan kolom 'wadmkc' (nama kecamatan) → 17 polygon kecamatan.

 Input:
   data/geojson/jumlah_jiwa_miskin_2024_1.json   (86 desa/kelurahan)
 Output:
   outputs/batas_kecamatan_sleman.geojson          (17 kecamatan, EPSG:4326)
==============================================================================
"""

import os
import json
import geopandas as gpd

# -- Paths -------------------------------------------------------------------
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_DIR = os.path.join(BASE_DIR, "data", "geojson")
OUTPUT_DIR = os.path.join(BASE_DIR, "outputs")
os.makedirs(OUTPUT_DIR, exist_ok=True)

INPUT_PATH  = os.path.join(DATA_DIR, "jumlah_jiwa_miskin_2024_1.json")
OUTPUT_PATH = os.path.join(OUTPUT_DIR, "batas_kecamatan_sleman.geojson")

# -- Load desa/kelurahan GeoJSON ----------------------------------------------
print("[1/2] Loading desa/kelurahan GeoJSON...")
gdf_desa = gpd.read_file(INPUT_PATH)
print(f"      {len(gdf_desa)} desa/kelurahan")
print(f"      CRS: {gdf_desa.crs}")
print(f"      Kolom: {list(gdf_desa.columns)}")

# -- Dissolve by kecamatan (wadmkc) ------------------------------------------
print("[2/2] Dissolving by wadmkc → kecamatan...")

# Pertahankan beberapa kolom administratif yang relevan
keep_cols = ["wadmkc", "wadmkk", "wadmpr", "wadmkd"]
available_keep = [c for c in keep_cols if c in gdf_desa.columns]

# Aggregate: ambil first value untuk kolom admin, sum untuk jumlah jiwa miskin
agg_dict = {}
for col in available_keep:
    agg_dict[col] = "first"
if "jmljwmskin" in gdf_desa.columns:
    agg_dict["jmljwmskin"] = "sum"

# Dissolve: wadmkc jadi index, lalu pindahkan ke kolom
gdf_kec = gdf_desa.dissolve(by="wadmkc", aggfunc=agg_dict)
gdf_kec["kecamatan"] = gdf_kec.index  # simpan nama kecamatan sebagai kolom
gdf_kec = gdf_kec.reset_index(drop=True)

print(f"      {len(gdf_kec)} kecamatan hasil dissolve")
print(f"      Nama kecamatan: {sorted(gdf_kec['kecamatan'].tolist())}")

# -- Reproject ke WGS84 (EPSG:4326) jika belum --------------------------------
if gdf_kec.crs is not None and gdf_kec.crs.to_epsg() != 4326:
    gdf_kec = gdf_kec.to_crs(epsg=4326)
    print(f"      Reprojected ke EPSG:4326")
elif gdf_kec.crs is None:
    gdf_kec.set_crs(epsg=4326, inplace=True)
    print(f"      CRS di-set ke EPSG:4326 (default)")

# -- Simplifikasi properti (buang kolom tidak perlu) --------------------------
# Hanya simpan kolom esensial
output_cols = ["kecamatan", "wadmkk", "jmljwmskin", "geometry"]
output_cols = [c for c in output_cols if c in gdf_kec.columns]
gdf_out = gdf_kec[output_cols].copy()

# Rename kolom agar lebih bersih
rename_map = {
    "wadmkk": "kabupaten",
    "jmljwmskin": "jumlah_jiwa_miskin",
}
gdf_out.rename(columns={k: v for k, v in rename_map.items() if k in gdf_out.columns}, inplace=True)

# -- Simpan GeoJSON -----------------------------------------------------------
gdf_out.to_file(OUTPUT_PATH, driver="GeoJSON")
print(f"\n  -> GeoJSON disimpan: {OUTPUT_PATH}")
print(f"  -> Jumlah kecamatan : {len(gdf_out)}")
print(f"  -> Properti         : {[c for c in gdf_out.columns if c != 'geometry']}")
print(f"  -> CRS              : EPSG:4326 (WGS84)")

# -- Preview ------------------------------------------------------------------
print(f"\n  Preview:")
for _, row in gdf_out.iterrows():
    geom_type = row.geometry.geom_type
    n_rings = len(row.geometry.geoms) if geom_type == "MultiPolygon" else 1
    print(f"    {row['kecamatan']:<16}  {geom_type} ({n_rings} parts)")
