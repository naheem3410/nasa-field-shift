import pandas as pd

from .cache import SqlStore, make_cache
from .config import Settings
from .regions import Regions
from .sources import AfspProfiles, IsdaSamples, IsdaSoilApi, NigeriaSoilMap


class SoilExplorer:
    def __init__(self, settings=None, session=None, cache=None):
        self.settings = settings or Settings.load()
        self.store = SqlStore(self.settings.database_url)
        self.regions = Regions(self.settings.boundaries_path)
        if cache is None:
            cache = make_cache(self.settings.redis_url, self.settings.redis_ttl)
        self.cache = cache
        args = (self.settings, self.store, self.regions)
        self.afsp = AfspProfiles(*args)
        self.samples = IsdaSamples(*args)
        self.soil_map = NigeriaSoilMap(*args)
        self.api = IsdaSoilApi(self.regions, self.cache, session=session)

    def tables(self, refresh=False):
        return {
            "afsp": self.afsp.frame(refresh),
            "isda_samples": self.samples.frame(refresh),
            "soil_map": self.soil_map.frame(refresh),
        }

    def overview(self):
        rows = []
        for name, src in (("afsp", self.afsp), ("isda_samples", self.samples), ("soil_map", self.soil_map)):
            df = src.frame()
            rows.append({
                "source": name,
                "rows": len(df),
                "columns": df.shape[1],
                "states": int(df["state"].nunique()),
                "note": src.note,
            })
        rows.append({"source": "isdasoil_api", "rows": None, "columns": None, "states": None,
                     "note": "per point, call at_point or api.point"})
        return pd.DataFrame(rows).set_index("source")

    def at_point(self, lat, lon, radius_km=25.0, properties=None, include_api=True):
        where = self.regions.locate(lat, lon)
        out = {
            "location": {
                "lat": float(lat), "lon": float(lon), **where,
                "nearest_afsp_km": round(self.afsp.nearest_km(lat, lon), 1),
                "nearest_isda_sample_km": round(self.samples.nearest_km(lat, lon), 1),
            },
            "afsp": self.afsp.near(lat, lon, radius_km),
            "isda_samples": self.samples.near(lat, lon, radius_km),
            "soil_map": self.soil_map.at_point(lat, lon),
        }
        if include_api:
            out["isdasoil"] = self.api.point(lat, lon, properties=properties)
        return out
