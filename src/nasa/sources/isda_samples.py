import pandas as pd

from .base import PointSource

FRONT = ["lat", "lon", "state", "geopolitical_zone", "state_distance_km"]


class IsdaSamples(PointSource):
    table = "isda_samples_nigeria"

    def inputs(self):
        return [self.settings.isda_csv]

    def build(self):
        raw = pd.read_csv(self.settings.isda_csv, encoding="utf-8-sig")
        raw = raw.rename(columns={"longitude": "lon", "latitude": "lat"})
        tagged = self.regions.tag_points(raw, max_km=0.0)
        inside = tagged["state"].notna()
        df = tagged[inside].reset_index(drop=True)
        df = df[FRONT + [c for c in df.columns if c not in FRONT]]
        note = {
            "rows_in_file": int(len(raw)),
            "rows_inside_nigeria": int(inside.sum()),
            "rows_dropped_outside_nigeria": int((~inside).sum()),
            "state_tolerance_km": 0.0,
        }
        return df, note

    def metadata(self):
        path = self.settings.isda_csv.with_name("iSDA_soil_metadata.csv")
        return pd.read_csv(path, encoding="utf-8-sig")
