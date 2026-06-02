import pandas as pd
import geopandas as gpd
from pathlib import Path

# Paths
BASE_DIR = Path(__file__).resolve().parent.parent
CSV_PATH = BASE_DIR / "data" / "data_final.csv"
GEOJSON_PATH = BASE_DIR / "data" / "geojson" / "jumlah_jiwa_miskin_2024_1.json"
OUTPUT_PATH = BASE_DIR / "data" / "data_final_with_centroid.csv"

# Load CSV data
df = pd.read_csv(CSV_PATH)

# Load GeoJSON data kecamatan (desa/kelurahan level)
gdf = gpd.read_file(GEOJSON_PATH)

# Dissolve geometries by kecamatan name (wadmkc)
kecamatan_dissolved = gdf.dissolve(by="wadmkc")

# Calculate centroid in the original projected CRS for accuracy
kecamatan_dissolved["centroid"] = kecamatan_dissolved.centroid

# Create GeoDataFrame from centroids and reproject to WGS84 (EPSG:4326)
centroids_gdf = gpd.GeoDataFrame(
    kecamatan_dissolved, geometry="centroid", crs=kecamatan_dissolved.crs
).to_crs(epsg=4326)

# Extract longitude and latitude
centroids_gdf["long"] = centroids_gdf.geometry.x
centroids_gdf["lat"] = centroids_gdf.geometry.y

# Prepare centroid dataframe for merging
centroid_df = centroids_gdf[["long", "lat"]].reset_index()
centroid_df.rename(columns={"wadmkc": "kecamatan"}, inplace=True)

# Merge with original CSV on kecamatan name
# Ensure consistent string formatting for matching
df["kecamatan"] = df["kecamatan"].astype(str).str.strip()
centroid_df["kecamatan"] = centroid_df["kecamatan"].astype(str).str.strip()

df_merged = df.merge(centroid_df, on="kecamatan", how="left")

# Check for unmatched kecamatan
unmatched = df_merged[df_merged["long"].isna()]["kecamatan"].tolist()
if unmatched:
    print(f"Warning: Could not find centroid for kecamatan: {unmatched}")
else:
    print("All kecamatan matched successfully.")

# Save to new CSV
df_merged.to_csv(OUTPUT_PATH, index=False)
print(f"Saved data with centroid to: {OUTPUT_PATH}")
print(df_merged[["kecamatan", "long", "lat"]].head())
