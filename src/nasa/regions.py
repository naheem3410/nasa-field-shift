import logging
from pathlib import Path

import geopandas as gpd
import pandas as pd
import requests

from .geo import AREA_CRS, METRIC_CRS

log = logging.getLogger("nasa")

BOUNDARY_URL = (
    "https://media.githubusercontent.com/media/wmgeolab/geoBoundaries/main/"
    "releaseData/gbOpen/NGA/ADM1/geoBoundaries-NGA-ADM1.geojson"
)

RENAMES = {"Abuja Federal Capital Territory": "Federal Capital Territory"}

GROUPS = {
    "North Central": ["Benue", "Federal Capital Territory", "Kogi", "Kwara", "Nasarawa", "Niger", "Plateau"],
    "North East": ["Adamawa", "Bauchi", "Borno", "Gombe", "Taraba", "Yobe"],
    "North West": ["Jigawa", "Kaduna", "Kano", "Katsina", "Kebbi", "Sokoto", "Zamfara"],
    "South East": ["Abia", "Anambra", "Ebonyi", "Enugu", "Imo"],
    "South South": ["Akwa Ibom", "Bayelsa", "Cross River", "Delta", "Edo", "Rivers"],
    "South West": ["Ekiti", "Lagos", "Ogun", "Ondo", "Osun", "Oyo"],
}
ZONES = {state: zone for zone, states in GROUPS.items() for state in states}


class OutsideNigeria(ValueError):
    pass


class Regions:
    def __init__(self, path):
        self.path = Path(path)
        if not self.path.exists():
            self._download()
        self.states = self._load()
        self._metric = self.states.to_crs(METRIC_CRS)
        minx, miny, maxx, maxy = self.states.total_bounds
        self._box = (minx - 0.5, miny - 0.5, maxx + 0.5, maxy + 0.5)

    def _download(self):
        log.info("downloading state boundaries to %s", self.path)
        resp = requests.get(BOUNDARY_URL, timeout=120)
        resp.raise_for_status()
        if not resp.content.startswith(b"{") or len(resp.content) < 100000:
            raise RuntimeError("boundary download did not return a geojson file")
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.path.write_bytes(resp.content)

    def _load(self):
        raw = gpd.read_file(self.path).to_crs(4326)
        raw["state"] = raw["shapeName"].map(lambda n: RENAMES.get(n, n))
        found, expected = set(raw["state"]), set(ZONES)
        if found != expected:
            raise ValueError(f"state names differ, missing {sorted(expected - found)}, extra {sorted(found - expected)}")
        raw["geopolitical_zone"] = raw["state"].map(ZONES)
        raw["geometry"] = raw.geometry.make_valid()
        return raw[["state", "geopolitical_zone", "geometry"]].sort_values("state").reset_index(drop=True)

    def signature(self):
        st = self.path.stat()
        return f"{self.path.name}:{st.st_size}:{st.st_mtime_ns}"

    def tag_points(self, df, lat="lat", lon="lon", max_km=0.0):
        out = df.drop(columns=["state", "geopolitical_zone", "state_distance_km"], errors="ignore")
        lats = pd.to_numeric(out[lat], errors="coerce")
        lons = pd.to_numeric(out[lon], errors="coerce")
        minx, miny, maxx, maxy = self._box
        ok = lons.between(minx, maxx) & lats.between(miny, maxy)
        pts = gpd.GeoDataFrame(
            geometry=gpd.points_from_xy(lons[ok], lats[ok]), index=out.index[ok], crs=4326
        ).to_crs(METRIC_CRS)
        hit = gpd.sjoin_nearest(
            pts, self._metric, how="left", max_distance=max(max_km * 1000, 1.0), distance_col="dist_m"
        )
        hit = hit.sort_values("dist_m")
        hit = hit[~hit.index.duplicated(keep="first")]
        tags = pd.DataFrame({
            "state": hit["state"],
            "geopolitical_zone": hit["geopolitical_zone"],
            "state_distance_km": (hit["dist_m"] / 1000).round(2),
        }).reindex(out.index)
        return pd.concat([out, tags], axis=1)

    def locate(self, lat, lon):
        one = self.tag_points(pd.DataFrame({"lat": [float(lat)], "lon": [float(lon)]}))
        state = one.at[0, "state"]
        if pd.isna(state):
            raise OutsideNigeria(f"({lat}, {lon}) is not inside the Nigeria state boundaries")
        return {"state": state, "geopolitical_zone": one.at[0, "geopolitical_zone"]}

    def split_polygons(self, gdf):
        base = gdf.to_crs(4326).copy()
        base["polygon_km2"] = base.to_crs(AREA_CRS).area / 1e6
        pieces = gpd.overlay(base, self.states, how="intersection", keep_geom_type=True)
        pieces = pieces[~pieces.geometry.is_empty].copy()
        pieces["piece_km2"] = pieces.to_crs(AREA_CRS).area / 1e6
        pieces["piece_share"] = pieces["piece_km2"] / pieces["polygon_km2"]
        return pieces.reset_index(drop=True)
