"""
fetchers/gee.py
---------------
Google Earth Engine data acquisition module.
Encapsulates authentication, image collection building, and spatial reduction
into a single reusable class.
"""
import json
from pathlib import Path
from typing import Optional

import ee
import pandas as pd

try:
    from .fetchers import config
    from utils import LOG, add_centroid_coords, add_distance_to_center, load_grid, save_csv
except ImportError:  # pragma: no cover
    import sys, pathlib
    sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[2]))
    from .fetchers import config
    from utils import LOG, add_centroid_coords, add_distance_to_center, load_grid, save_csv


class GEEFetcher:
    """
    Research-grade fetcher for remote-sensing features from GEE.

    Parameters
    ----------
    project : str
        GEE cloud project ID.
    year : int
        Acquisition year.
    """

    def __init__(self, project: str = config.GEE_PROJECT, year: int = config.GEE_YEAR):
        self.project = project
        self.year = year
        self._initialised = False

    # -------------------------------------------------------------------------
    # Authentication
    # -------------------------------------------------------------------------
    def initialise(self) -> None:
        """Initialise the Earth Engine session (idempotent)."""
        if self._initialised:
            return
        try:
            ee.Initialize(project=self.project)
            LOG.info("GEE initialised (project=%s)", self.project)
        except Exception as exc:
            LOG.warning("GEE auto-init failed (%s); triggering authentication...", exc)
            ee.Authenticate()
            ee.Initialize(project=self.project)
            LOG.info("GEE authenticated & initialised")
        self._initialised = True

    # -------------------------------------------------------------------------
    # Grid helpers
    # -------------------------------------------------------------------------
    @staticmethod
    def load_grid_as_fc(path: Path = config.GRID_PATH) -> ee.FeatureCollection:
        """Read the local GeoJSON and cast it to an ee.FeatureCollection."""
        if not path.exists():
            raise FileNotFoundError(f"Grid not found: {path}")
        with open(path, "r", encoding="utf-8") as f:
            geojson = json.load(f)
        fc = ee.FeatureCollection(geojson)
        n = fc.size().getInfo()
        LOG.info("GEE grid loaded: %d cells", n)
        return fc

    # -------------------------------------------------------------------------
    # Image helpers
    # -------------------------------------------------------------------------
    def _date_range(self) -> tuple:
        return f"{self.year}-01-01", f"{self.year}-12-31"

    @staticmethod
    def get_viirs_nl(year: int) -> ee.Image:
        """VIIRS DNB Annual mean night-light radiance."""
        img = (
            ee.ImageCollection("NOAA/VIIRS/DNB/MONTHLY_V1/VCMSLCFG")
            .filterDate(f"{year}-01-01", f"{year}-12-31")
            .select("avg_rad")
            .mean()
            .rename("night_light")
        )
        return img

    @staticmethod
    def get_modis_ndvi(start: str, end: str) -> ee.Image:
        """MOD13Q1 NDVI mean, scaled by 1e-4."""
        return (
            ee.ImageCollection("MODIS/061/MOD13Q1")
            .filterDate(start, end)
            .select("NDVI")
            .mean()
            .multiply(0.0001)
            .rename("ndvi")
        )

    @staticmethod
    def get_modis_lst(start: str, end: str) -> ee.Image:
        """MOD11A1 LST Day mean, scaled & converted to °C."""
        return (
            ee.ImageCollection("MODIS/061/MOD11A1")
            .filterDate(start, end)
            .select("LST_Day_1km")
            .mean()
            .multiply(0.02)
            .subtract(273.15)
            .rename("lst_celsius")
        )

    @staticmethod
    def get_landsat_ndbi(region: ee.Geometry, start: str, end: str) -> ee.Image:
        """
        Landsat-8 Collection-2 NDBI median.
        NDBI = (SWIR1 - NIR) / (SWIR1 + NIR)
        """
        def mask_landsat(image: ee.Image) -> ee.Image:
            qa = image.select("QA_PIXEL")
            cloud = qa.bitwiseAnd(1 << 3).eq(0)
            shadow = qa.bitwiseAnd(1 << 4).eq(0)
            return image.updateMask(cloud.And(shadow))

        def scale(image: ee.Image) -> ee.Image:
            scaled = image.select(["SR_B5", "SR_B6"]).multiply(0.0000275).add(-0.2)
            return image.addBands(scaled, None, True)

        coll = (
            ee.ImageCollection("LANDSAT/LC08/C02/T1_L2")
            .filterDate(start, end)
            .filterBounds(region)
            .map(mask_landsat)
            .map(scale)
        )

        ndbi = (
            coll.select(["SR_B6", "SR_B5"])
            .median()
            .normalizedDifference(["SR_B6", "SR_B5"])
            .rename("ndbi")
        )
        return ndbi

    # -------------------------------------------------------------------------
    # Reduction
    # -------------------------------------------------------------------------
    @staticmethod
    def _fc_to_dataframe(fc: ee.FeatureCollection) -> pd.DataFrame:
        """Convert an Earth Engine FeatureCollection to a tidy DataFrame."""
        info = fc.getInfo()
        features = info.get("features", [])
        if not features:
            LOG.warning("Empty FeatureCollection returned from GEE")
            return pd.DataFrame()

        rows = []
        for f in features:
            props = f.get("properties", {})
            rows.append(props)
        return pd.DataFrame(rows)

    # -------------------------------------------------------------------------
    # Public API
    # -------------------------------------------------------------------------
    def run(
        self,
        grid_path: Optional[Path] = None,
        out_path: Optional[Path] = None,
    ) -> pd.DataFrame:
        """
        Execute the full GEE acquisition workflow.

        Returns
        -------
        pd.DataFrame
            One row per grid cell with night_light, ndvi, ndbi, lst_celsius,
            dist_to_center_km, centroid_lon, centroid_lat.
        """
        self.initialise()

        grid_path = grid_path or config.GRID_PATH
        out_path = out_path or config.GEE_OUTPUT

        # --- 1. Load grid -----------------------------------------------------
        grid = self.load_grid_as_fc(grid_path)

        # --- 2. Add distance & centroid ---------------------------------------
        center = ee.Geometry.Point([config.CENTER_LON, config.CENTER_LAT])
        grid = grid.map(
            lambda f: f.set(
                "dist_to_center_km",
                f.centroid().distance(center).divide(1000),
            )
        )
        grid = grid.map(
            lambda f: f.set(
                "centroid_lon", f.geometry().centroid().coordinates().get(0)
            ).set(
                "centroid_lat", f.geometry().centroid().coordinates().get(1)
            )
        )
        LOG.info("Distance & centroid metadata attached")

        # --- 3. Build composite image -----------------------------------------
        start, end = self._date_range()
        nl = self.get_viirs_nl(self.year)
        ndvi = self.get_modis_ndvi(start, end)
        lst = self.get_modis_lst(start, end)
        ndbi = self.get_landsat_ndbi(grid.geometry(), start, end)

        composite = nl.addBands(ndvi).addBands(lst).addBands(ndbi)
        LOG.info("Composite image built (bands: %s)", composite.bandNames().getInfo())

        # --- 4. Reduce to grid cells ------------------------------------------
        LOG.info("Reducing image to grid cells (scale=%dm, crs=%s)...",
                 config.GEE_REDUCE_SCALE, config.GEE_REDUCE_CRS)
        reduced = composite.reduceRegions(
            collection=grid,
            reducer=ee.Reducer.mean(),
            scale=config.GEE_REDUCE_SCALE,
            crs=config.GEE_REDUCE_CRS,
        )

        # --- 5. Convert & tidy ------------------------------------------------
        df = self._fc_to_dataframe(reduced)
        if df.empty:
            raise RuntimeError("GEE reduction returned zero rows.")

        # Reorder / select columns for consistency
        expected = [
            config.GRID_ID_COL,
            "centroid_lon",
            "centroid_lat",
            "dist_to_center_km",
            "night_light",
            "ndvi",
            "ndbi",
            "lst_celsius",
        ]
        available = [c for c in expected if c in df.columns]
        df = df[available]

        # --- 6. Persist -------------------------------------------------------
        save_csv(df, out_path)
        LOG.info("GEE pipeline complete -- %d rows", len(df))
        return df


# Allow running as a standalone script
if __name__ == "__main__":
    fetcher = GEEFetcher()
    fetcher.run()
