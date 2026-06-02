import geopandas as gpd
from shapely.geometry import box
import os

# Paths
input_path = os.path.join('data', 'gadm41_IDN_2.json')
output_path = 'kotayogyakarta_grid_2km.geojson'

print("Reading geo data...")
gdf = gpd.read_file(input_path)

# Filter Kota Yogyakarta (Yogyakarta)
kotayogyakarta = gdf[(gdf['NAME_2'] == 'KotaYogyakarta') & (gdf['NAME_1'] == 'Yogyakarta')].copy()
print(f"Kota Yogyakarta feature found: {len(kotayogyakarta)}")

if len(kotayogyakarta) == 0:
    raise ValueError("Kota Yogyakarta tidak ditemukan di data.")

# Pastikan CRS WGS84
kotayogyakarta = kotayogyakarta.set_crs(epsg=4326, allow_override=True)

# Proyeksikan ke EPSG:3857 (Web Mercator) untuk grid metrik yang presisi ~1 km
kotayogyakarta_3857 = kotayogyakarta.to_crs(epsg=3857)

# Bounding box metrik
minx, miny, maxx, maxy = kotayogyakarta_3857.total_bounds
cell_size = 2000  # 1 km

# Generate grid cells
geometries = []
x = minx
while x < maxx:
    y = miny
    while y < maxy:
        geometries.append(box(x, y, x + cell_size, y + cell_size))
        y += cell_size
    x += cell_size

grid_3857 = gpd.GeoDataFrame({'geometry': geometries}, crs='EPSG:3857')

# Clip ke batas Kota Yogyakarta agar hanya grid yang overlap/intersect
grid_3857 = gpd.clip(grid_3857, kotayogyakarta_3857)

# Kembalikan ke EPSG:4326 untuk output GeoJSON standar
grid_4326 = grid_3857.to_crs(epsg=4326)

# Reset index dan tambah grid_id
grid_4326 = grid_4326.reset_index(drop=True)
grid_4326['grid_id'] = [f"KOT_{i+1:05d}" for i in range(len(grid_4326))]

# Hanya simpan kolom yang diminta
grid_4326 = grid_4326[['grid_id', 'geometry']]

# Simpan GeoJSON
grid_4326.to_file(output_path, driver='GeoJSON')
print(f"Selesai. Total grid: {len(grid_4326)}. Disimpan ke: {output_path}")
