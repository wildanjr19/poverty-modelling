import geopandas as gpd

gdf = gpd.read_file("sleman_grid_2km.geojson")

# Simpan ke shapefile
gdf.to_file("sleman_grid_2km.shp")