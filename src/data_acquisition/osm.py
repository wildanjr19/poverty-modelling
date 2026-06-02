"""
fetchers/osm.py
---------------
OpenStreetMap POI acquisition via the Overpass API.
Queries node + way objects per amenity/shop tag and performs a spatial join
against the master grid to produce count features per cell.
"""
import time
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import geopandas as gpd
import pandas as pd
import requests
from shapely.geometry import Point

try:
    from .fetchers import config
    from utils import LOG, load_grid, save_csv
except ImportError:  # pragma: no cover
    import sys, pathlib
    sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[2]))
    from .fetchers import config
    from utils import LOG, load_grid, save_csv


class OSMFetcher:
    """
    Research-grade fetcher for OSM Points-of-Interest.

    Parameters
    ----------
    tags : dict
        Mapping ``column_name -> (osm_key, osm_value)``.
    overpass_url : str
        Overpass API endpoint.
    timeout : int
        Query timeout in seconds.
    delay : float
        Sleep duration between successive queries (rate limiting).
    """

    def __init__(
        self,
        tags: Optional[Dict[str, Tuple[str, str]]] = None,
        overpass_url: str = config.OVERPASS_URL,
        timeout: int = config.OVERPASS_TIMEOUT,
        delay: float = config.OVERPASS_DELAY_S,
    ):
        self.tags = tags or config.OSM_POI_TAGS
        self.overpass_url = overpass_url
        self.timeout = timeout
        self.delay = delay

    # -------------------------------------------------------------------------
    # Overpass query
    # -------------------------------------------------------------------------
    def query_overpass(
        self,
        bbox: Tuple[float, float, float, float],
        key: str,
        value: str,
    ) -> List[dict]:
        """
        Query Overpass for nodes and ways inside a bounding box.

        bbox format: (south, west, north, east)
        """
        query = f"""
        [out:json][timeout:{self.timeout}];
        (
          node["{key}"="{value}"]({bbox[0]},{bbox[1]},{bbox[2]},{bbox[3]});
          way["{key}"="{value}"]({bbox[0]},{bbox[1]},{bbox[2]},{bbox[3]});
        );
        out center;
        """
        try:
            resp = requests.post(
                self.overpass_url,
                data={"data": query},
                headers={
                    "Content-Type": "application/x-www-form-urlencoded; charset=UTF-8",
                    "Accept": "application/json",
                },
                timeout=self.timeout + 10,
            )
            resp.raise_for_status()
            data = resp.json()
            elements = data.get("elements", [])
            LOG.info("Overpass '%s=%s' returned %d elements", key, value, len(elements))
            return elements
        except requests.RequestException as exc:
            LOG.error("Overpass query failed for '%s=%s': %s", key, value, exc)
            raise

    # -------------------------------------------------------------------------
    # Spatial processing
    # -------------------------------------------------------------------------
    @staticmethod
    def _elements_to_gdf(elements: List[dict]) -> gpd.GeoDataFrame:
        """Convert Overpass elements to a GeoDataFrame of Points."""
        points: List[Point] = []
        for el in elements:
            if el.get("type") == "node":
                points.append(Point(el["lon"], el["lat"]))
            elif el.get("type") == "way" and "center" in el:
                c = el["center"]
                points.append(Point(c["lon"], c["lat"]))
            # relations are ignored for POI counting
        gdf = gpd.GeoDataFrame({"geometry": points}, crs=config.GRID_CRS)
        return gdf

    @staticmethod
    def count_poi_per_grid(
        grid_gdf: gpd.GeoDataFrame,
        poi_gdf: gpd.GeoDataFrame,
    ) -> gpd.GeoDataFrame:
        """
        Spatial join POI points into grid polygons and count per cell.
        Returns a DataFrame with grid_id and count.
        """
        if poi_gdf.empty:
            grid_gdf["count"] = 0
            return grid_gdf[[config.GRID_ID_COL, "count"]]

        joined = gpd.sjoin(poi_gdf, grid_gdf, predicate="within", how="left")
        counts = (
            joined.groupby(config.GRID_ID_COL)
            .size()
            .reset_index(name="count")
        )
        result = grid_gdf[[config.GRID_ID_COL]].merge(
            counts, on=config.GRID_ID_COL, how="left"
        )
        result["count"] = result["count"].fillna(0).astype(int)
        return result

    # -------------------------------------------------------------------------
    # Public API
    # -------------------------------------------------------------------------
    def run(
        self,
        grid_path: Optional[Path] = None,
        out_path: Optional[Path] = None,
    ) -> pd.DataFrame:
        """
        Execute the full OSM acquisition workflow.

        Returns
        -------
        pd.DataFrame
            One row per grid cell with ``*_count`` columns.
        """
        grid_path = grid_path or config.GRID_PATH
        out_path = out_path or config.OSM_OUTPUT

        grid = load_grid(grid_path)
        total_bounds = grid.total_bounds  # minx, miny, maxx, maxy
        bbox = (total_bounds[1], total_bounds[0], total_bounds[3], total_bounds[2])
        LOG.info("Overpass bbox (S,W,N,E): %.5f, %.5f, %.5f, %.5f", *bbox)

        out_df = grid[[config.GRID_ID_COL]].copy()

        for col_name, (key, value) in self.tags.items():
            LOG.info("Querying OSM -> %s", col_name)
            elements = self.query_overpass(bbox, key, value)
            poi_gdf = self._elements_to_gdf(elements)
            counted = self.count_poi_per_grid(grid, poi_gdf)
            out_df[col_name] = counted["count"].values
            time.sleep(self.delay)

        save_csv(out_df, out_path)
        LOG.info("OSM pipeline complete -- %d rows, %d POI layers", len(out_df), len(self.tags))
        return out_df


# Allow running as a standalone script
if __name__ == "__main__":
    fetcher = OSMFetcher()
    fetcher.run()
