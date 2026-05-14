"""
utils.py
--------
Shared helpers used across the pipeline.
"""
import logging
import sys
from pathlib import Path
from typing import List, Union

import geopandas as gpd
import pandas as pd

from . import config

# -----------------------------------------------------------------------------
# Logging
# -----------------------------------------------------------------------------
def setup_logging(level: int = logging.INFO) -> logging.Logger:
    """Configure a consistent logger for the whole pipeline."""
    logger = logging.getLogger("sleman_pipeline")
    logger.setLevel(level)
    if not logger.handlers:
        handler = logging.StreamHandler(sys.stdout)
        handler.setLevel(level)
        fmt = logging.Formatter("[%(levelname)s] %(name)s :: %(message)s")
        handler.setFormatter(fmt)
        logger.addHandler(handler)
    return logger


LOG = setup_logging()


# -----------------------------------------------------------------------------
# I/O helpers
# -----------------------------------------------------------------------------
def load_grid(path: Union[str, Path] = config.GRID_PATH) -> gpd.GeoDataFrame:
    """Load the master grid GeoJSON and normalise CRS."""
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"Grid file not found: {path}")
    gdf = gpd.read_file(path)
    gdf = gdf.to_crs(config.GRID_CRS)
    if config.GRID_ID_COL not in gdf.columns:
        raise ValueError(f"Grid must contain a '{config.GRID_ID_COL}' column.")
    # Ensure consistent column set
    gdf = gdf[[config.GRID_ID_COL, "geometry"]].copy()
    LOG.info("Loaded %d grid cells from %s", len(gdf), path)
    return gdf


def save_csv(df: pd.DataFrame, path: Union[str, Path]) -> None:
    """Write a DataFrame to CSV, creating parent dirs if needed."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(path, index=False)
    LOG.info("Saved CSV: %s (rows=%d, cols=%d)", path, len(df), len(df.columns))


def load_csv(path: Union[str, Path]) -> pd.DataFrame:
    """Load a CSV and verify that 'grid_id' exists."""
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"CSV not found: {path}")
    df = pd.read_csv(path)
    if config.GRID_ID_COL not in df.columns:
        raise ValueError(f"CSV must contain '{config.GRID_ID_COL}': {path}")
    return df


# -----------------------------------------------------------------------------
# Merge / Validation
# -----------------------------------------------------------------------------
def merge_feature_csvs(
    paths: List[Union[str, Path]],
    on: str = config.GRID_ID_COL,
    how: str = "left",
) -> pd.DataFrame:
    """
    Merge multiple feature CSVs (all keyed by `grid_id`) into a single
    DataFrame. Duplicate columns (other than the key) raise an error.
    """
    if not paths:
        raise ValueError("No CSV paths provided for merge.")

    base = load_csv(paths[0])
    LOG.info("Merging %d feature files on '%s'", len(paths), on)

    for p in paths[1:]:
        df = load_csv(p)
        # Detect column collisions (excluding merge key)
        overlap = set(base.columns) & set(df.columns) - {on}
        if overlap:
            raise ValueError(
                f"Column collision when merging {p}: {overlap}. "
                "Ensure each fetcher writes mutually exclusive feature columns."
            )
        base = base.merge(df, on=on, how=how)

    return base


def enforce_schema(df: pd.DataFrame, schema: List[str] = config.FINAL_SCHEMA) -> pd.DataFrame:
    """
    Reorder columns to match the desired schema. Missing columns are filled
    with NaN and a warning is logged; extra columns are dropped.
    """
    missing = [c for c in schema if c not in df.columns]
    if missing:
        LOG.warning("Schema missing columns (filled with NaN): %s", missing)
        for c in missing:
            df[c] = pd.NA

    extra = [c for c in df.columns if c not in schema]
    if extra:
        LOG.info("Dropping extra columns not in schema: %s", extra)

    return df[[c for c in schema if c in df.columns]].copy()


# -----------------------------------------------------------------------------
# Geometry helpers
# -----------------------------------------------------------------------------
def add_distance_to_center(
    gdf: gpd.GeoDataFrame,
    lon: float = config.CENTER_LON,
    lat: float = config.CENTER_LAT,
    col_name: str = "dist_to_center_km",
) -> gpd.GeoDataFrame:
    """Add a column with haversine distance (km) from centroid to a point."""
    from shapely.geometry import Point

    center = Point(lon, lat)
    # Use geopandas distance in a projected CRS for accuracy, then convert back
    gdf_proj = gdf.to_crs(epsg=3857)
    center_proj = gpd.GeoSeries([center], crs="EPSG:4326").to_crs(epsg=3857).iloc[0]
    gdf_proj[col_name] = gdf_proj.centroid.distance(center_proj) / 1000.0
    gdf = gdf_proj.to_crs(config.GRID_CRS)
    LOG.info("Added '%s' (km) relative to (%.4f, %.4f)", col_name, lon, lat)
    return gdf


def add_centroid_coords(
    gdf: gpd.GeoDataFrame,
    lon_col: str = "centroid_lon",
    lat_col: str = "centroid_lat",
) -> gpd.GeoDataFrame:
    """Append centroid longitude / latitude columns."""
    gdf[lon_col] = gdf.centroid.x
    gdf[lat_col] = gdf.centroid.y
    return gdf
