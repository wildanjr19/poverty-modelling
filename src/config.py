"""
config.py
---------
Centralised configuration for the Sleman data pipeline.
All paths, parameters, and schema definitions live here so that
notebooks / scripts / tests share a single source of truth.

Usage:
    from src.config import DATA_DIR, MODEL_DATA_PATH, ...
"""
from pathlib import Path

# =============================================================================
# 1. PROJECT ROOT & DIRECTORY STRUCTURE
# =============================================================================
PROJECT_ROOT = Path(__file__).resolve().parent.parent

DATA_DIR   = PROJECT_ROOT / "data"
RAW_DIR    = DATA_DIR / "raw"
INTERIM_DIR = DATA_DIR / "interim"
GEOJSON_DIR = DATA_DIR / "geojson"
OUTPUT_DIR  = PROJECT_ROOT / "outputs"

# Ensure directories exist at import time (safe, idempotent)
for _d in (RAW_DIR, INTERIM_DIR, GEOJSON_DIR, OUTPUT_DIR):
    _d.mkdir(parents=True, exist_ok=True)

# =============================================================================
# 2. INPUT DATA – CSV
# =============================================================================
# Final dataset (kecamatan-level, no centroid)
MODEL_DATA_PATH = DATA_DIR / "data_final.csv"
# Final dataset WITH centroid lon/lat columns
MODEL_DATA_CENTROID_PATH = DATA_DIR / "data_final_with_centroid.csv"
# Raw merged dataset (grid-level)
FINAL_DATASET_PATH = RAW_DIR / "final_dataset.csv"

# =============================================================================
# 3. INPUT DATA – GeoJSON
# =============================================================================
ADMIN_DESA_GEOJSON    = GEOJSON_DIR / "jumlah_jiwa_miskin_2024_1.json"
ADMIN_KABUPATEN_GEOJSON = DATA_DIR / "gadm41_IDN_2.json"

# Grid GeoJSON per kabupaten
GRID_FILES = {
    "Bantul":         GEOJSON_DIR / "bantul_grid_2km.geojson",
    "Gunungkidul":    GEOJSON_DIR / "gunungkidul_grid_2km.geojson",
    "Kota Yogyakarta": GEOJSON_DIR / "kotayogyakarta_grid_2km.geojson",
    "Kulon Progo":    GEOJSON_DIR / "kulonprogo_grid_2km.geojson",
    "Sleman":         GEOJSON_DIR / "sleman_grid_2km.geojson",
}
# Default grid (Sleman only, for the main pipeline)
GRID_PATH = GRID_FILES["Sleman"]

# =============================================================================
# 4. OUTPUT PATHS
# =============================================================================
# Intermediates (GEE / OSM)
GEE_OUTPUT  = INTERIM_DIR / "features_gee.csv"
OSM_OUTPUT  = INTERIM_DIR / "features_osm.csv"

# Preprocessing outputs
BATAS_KECAMATAN_PATH = OUTPUT_DIR / "batas_kecamatan_sleman.geojson"

# Modelling outputs
GPR_PREDICTIONS_PATH  = OUTPUT_DIR / "gpr_best_predictions.csv"

# Visualisation outputs
VIZ_GRID_DIY_PATH         = OUTPUT_DIR / "visualisasi_grid_diy.png"
VIZ_MAP_KEMISKINAN_PATH   = OUTPUT_DIR / "map_kemiskinan_sleman.png"
VIZ_MAP_KEMISKINAN_HIRES  = OUTPUT_DIR / "map_kemiskinan_sleman_hires.png"
VIZ_SHAP_BEESWARM_PATH    = OUTPUT_DIR / "shap_beeswarm_best_gpr.png"
VIZ_SHAP_BAR_PATH         = OUTPUT_DIR / "shap_bar_best_gpr.png"

# =============================================================================
# 5. GRID / GEOMETRY
# =============================================================================
GRID_CRS    = "EPSG:4326"      # WGS84
GRID_CRS_METRIC = "EPSG:3857"  # Web Mercator (metric operations)
GRID_ID_COL = "grid_id"
GRID_CELL_SIZE_M = 2000        # 2 km grid cells

# Center of Yogyakarta (km distance metric)
CENTER_LON = 110.3648
CENTER_LAT = -7.8012

# GADM filter strings (Kabupaten-level)
GADM_NAME_1 = "Yogyakarta"
GADM_KABUPATEN_FILTERS = {
    "Sleman":          {"NAME_2": "Sleman"},
    "Bantul":          {"NAME_2": "Bantul"},
    "Gunungkidul":     {"NAME_2": "Gunungkidul"},
    "Kulon Progo":     {"NAME_2": "KulonProgo"},
    "Kota Yogyakarta":  {"NAME_2": "KotaYogyakarta"},
}

# Grid ID prefix per kabupaten
GRID_ID_PREFIX = {
    "Sleman":          "SLE",
    "Bantul":          "BTL",
    "Gunungkidul":     "GK",
    "Kulon Progo":     "KP",
    "Kota Yogyakarta":  "KOT",
}

# =============================================================================
# 6. GOOGLE EARTH ENGINE
# =============================================================================
GEE_PROJECT    = "project-a5d1a726-a49d-435d-a96"
GEE_YEAR       = 2023
GEE_START_DATE = f"{GEE_YEAR}-01-01"
GEE_END_DATE   = f"{GEE_YEAR}-12-31"

# reduceRegions parameters
GEE_REDUCE_SCALE = 500          # metres
GEE_REDUCE_CRS   = "EPSG:32749" # UTM zone 49S – covers Sleman

# -- Dataset & band constants ------------------------------------------------
# VIIRS Night Light
VIIRS_COLLECTION = "NOAA/VIIRS/DNB/MONTHLY_V1/VCMSLCFG"
VIIRS_BAND       = "avg_rad"
VIIRS_OUTPUT     = "night_light"

# MODIS NDVI
MODIS_NDVI_COLLECTION = "MODIS/061/MOD13Q1"
MODIS_NDVI_BAND       = "NDVI"
MODIS_NDVI_SCALE      = 0.0001
MODIS_NDVI_OUTPUT     = "ndvi"

# MODIS LST
MODIS_LST_COLLECTION = "MODIS/061/MOD11A1"
MODIS_LST_BAND       = "LST_Day_1km"
MODIS_LST_SCALE      = 0.02
MODIS_LST_KELVIN_OFFSET = 273.15
MODIS_LST_OUTPUT     = "lst_celsius"

# Landsat-8 NDBI
LANDSAT_COLLECTION   = "LANDSAT/LC08/C02/T1_L2"
LANDSAT_NIR_BAND     = "SR_B5"
LANDSAT_SWIR1_BAND   = "SR_B6"
LANDSAT_QA_BAND      = "QA_PIXEL"
LANDSAT_SCALE        = 0.0000275
LANDSAT_OFFSET       = -0.2
LANDSAT_CLOUD_BIT    = 3
LANDSAT_SHADOW_BIT   = 4
LANDSAT_NDBI_OUTPUT  = "ndbi"

# =============================================================================
# 7. OPENSTREETMAP / OVERPASS
# =============================================================================
OVERPASS_URL     = "https://overpass.kumi.systems/api/interpreter"
OVERPASS_TIMEOUT = 60
OVERPASS_DELAY_S = 1.5           # polite delay between queries

# POI definitions: column_name -> (osm_key, osm_value)
OSM_POI_TAGS = {
    "school_count":   ("amenity", "school"),
    "hospital_count": ("amenity", "hospital"),
    "bank_count":     ("amenity", "bank"),
    "market_count":   ("shop",    "supermarket"),
}

# =============================================================================
# 8. COLUMN NAME CONSTANTS (shared across modelling, spatial, viz)
# =============================================================================
TARGET_COL           = "persentase_penduduk_miskin"
KECAMATAN_COL        = "kecamatan"
LONG_COL             = "long"
LAT_COL              = "lat"

# Admin boundary columns (desa/kelurahan GeoJSON)
ADMIN_KEC_COL   = "wadmkc"   # nama kecamatan
ADMIN_KAB_COL   = "wadmkk"   # nama kabupaten
ADMIN_PROV_COL  = "wadmpr"   # nama provinsi
ADMIN_DESA_COL  = "wadmkd"   # nama desa/kelurahan
ADMIN_JIWA_COL  = "jmljwmskin"  # jumlah jiwa miskin

# Prediction columns (CSV output)
PRED_COL_AKTUAL    = "aktual_pct"
PRED_COL_PREDIKSI  = "prediksi_pct"
PRED_COL_APE       = "ape_pct"

# =============================================================================
# 9. OUTPUT SCHEMA (desired column order for the final grid dataset)
# =============================================================================
FINAL_SCHEMA = [
    GRID_ID_COL,
    VIIRS_OUTPUT,
    MODIS_NDVI_OUTPUT,
    LANDSAT_NDBI_OUTPUT,
    MODIS_LST_OUTPUT,
    "dist_to_center_km",
    "school_count",
    "hospital_count",
    "bank_count",
    "market_count",
    "centroid_lon",
    "centroid_lat",
]

# =============================================================================
# 10. SPATIAL ANALYSIS
# =============================================================================
MORAN_K_NEIGHBORS   = 4     # k for KNN spatial weights
MORAN_PERMUTATIONS  = 999   # n permutations for Moran's I
MORAN_SIGNIFICANCE  = 0.05  # alpha threshold

# =============================================================================
# 11. MODELLING HYPERPARAMETERS (shared defaults)
# =============================================================================
RANDOM_STATE = 42

# GPR defaults
GPR_N_RESTARTS    = 10
GPR_LENGTH_SCALE  = 1.0
GPR_NOISE_LEVEL   = 0.1
GPR_ALPHA_DEFAULT = 1e-10

# Ridge
RIDGE_ALPHA = 1.0

# Lasso
LASSO_ALPHA   = 0.1
LASSO_MAX_ITER = 10_000

# SVR
SVR_C       = 10
SVR_EPSILON = 0.5

# Random Forest
RF_N_ESTIMATORS   = 100
RF_MAX_DEPTH      = 3
RF_MIN_SAMPLES_LEAF = 3

# XGBoost
XGB_N_ESTIMATORS  = 50
XGB_MAX_DEPTH     = 2
XGB_LEARNING_RATE = 0.1
XGB_REG_LAMBDA    = 5
XGB_REG_ALPHA     = 1

# Cook's Distance weighting
COOK_EPSILON       = 1e-9      # numerical stability
COOK_SIGMA_BASE    = 1.0       # base noise for heteroskedastic GPR

# Models that don't support sample_weight
NO_SAMPLE_WEIGHT_MODELS = {"Gaussian Process", "SVR (RBF)"}

# =============================================================================
# 12. VISUALISATION
# =============================================================================
# Kabupaten colour map
KABUPATEN_COLORS = {
    "Bantul":         "#e41a1c",
    "Gunungkidul":    "#377eb8",
    "Kota Yogyakarta": "#4daf4a",
    "Kulon Progo":    "#984ea3",
    "Sleman":         "#ff7f00",
}

# Choropleth colour maps
CMAP_AKTUAL = "YlOrRd"
CMAP_PRED   = "YlOrRd"
CMAP_APE    = "RdYlGn_r"
