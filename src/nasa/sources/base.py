import json
import logging

import numpy as np

from ..geo import haversine_km

log = logging.getLogger("nasa")

# bump this to force every cached table to rebuild
CACHE_VERSION = "1"


class FileSource:
    table = ""

    def __init__(self, settings, store, regions):
        self.settings = settings
        self.store = store
        self.regions = regions
        self._frame = None
        self._note = None

    def inputs(self):
        raise NotImplementedError

    def build(self):
        raise NotImplementedError

    def to_store(self, df):
        return df

    def from_store(self, df):
        return df

    def signature(self):
        parts = [CACHE_VERSION, self.regions.signature()]
        for path in self.inputs():
            if not path.exists():
                raise FileNotFoundError(f"missing input file: {path}")
            st = path.stat()
            parts.append(f"{path.name}:{st.st_size}:{st.st_mtime_ns}")
        return "|".join(parts)

    def frame(self, refresh=False):
        if self._frame is None or refresh:
            self._frame = self._load(refresh)
        return self._frame.copy()

    def _load(self, refresh):
        sig = self.signature()
        if not refresh and self.store.is_fresh(self.table, sig):
            self._note = json.loads(self.store.note(self.table) or "{}")
            return self.from_store(self.store.read(self.table))
        df, note = self.build()
        self.store.write(self.table, self.to_store(df), sig, json.dumps(note))
        self._note = note
        return df

    @property
    def note(self):
        if self._note is None:
            self.frame()
        return self._note


class PointSource(FileSource):
    def near(self, lat, lon, radius_km=25.0):
        df = self.frame()
        dist = haversine_km(lat, lon, df["lat"].to_numpy(dtype=float), df["lon"].to_numpy(dtype=float))
        out = df.assign(distance_km=dist.round(2))
        return out[out["distance_km"] <= radius_km].sort_values("distance_km")

    def nearest_km(self, lat, lon):
        df = self.frame()
        dist = haversine_km(lat, lon, df["lat"].to_numpy(dtype=float), df["lon"].to_numpy(dtype=float))
        return float(np.nanmin(dist))
