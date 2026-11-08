import pyogrio

from .base import PointSource

FRONT = ["profileid", "layerid", "lat", "lon", "state", "geopolitical_zone", "state_distance_km"]


class AfspProfiles(PointSource):
    table = "afsp_nigeria"
    state_tolerance_km = 10.0

    def inputs(self):
        folder = self.settings.afsp_dir
        return [folder / "AfSP012Qry_Profiles.dbf", folder / "AfSP012Qry_Layers.dbf"]

    def _read(self, path):
        df = pyogrio.read_dataframe(path, where="ProfileID LIKE 'NG%'", read_geometry=False)
        df.columns = [c.lower() for c in df.columns]
        return df

    def build(self):
        profile_file, layer_file = self.inputs()
        profiles = self._read(profile_file)
        layers = self._read(layer_file)
        df = layers.merge(profiles, on="profileid", how="left", suffixes=("", "_profile"), validate="many_to_one")
        df, missing = self._clean(df)

        no_coords = df["x_londd"].isna() | df["y_latdd"].isna() | ((df["x_londd"] == 0) & (df["y_latdd"] == 0))
        df["lon"] = df["x_londd"].mask(no_coords)
        df["lat"] = df["y_latdd"].mask(no_coords)
        df = self.regions.tag_points(df, max_km=self.state_tolerance_km)

        front = [c for c in FRONT if c in df.columns]
        df = df[front + [c for c in df.columns if c not in front]]
        note = {
            "profiles": int(profiles["profileid"].nunique()),
            "profiles_without_layers": int((~profiles["profileid"].isin(layers["profileid"])).sum()),
            "layers": int(len(layers)),
            "layers_without_coordinates": int(no_coords.sum()),
            "layers_without_state": int(df["state"].isna().sum()),
            "missing_markers_set_to_nan": missing,
            "state_tolerance_km": self.state_tolerance_km,
        }
        return df, note

    def _clean(self, df):
        num = df.select_dtypes("number").columns
        txt = df.select_dtypes(exclude="number").columns
        num_hits = int((df[num] == -9999).sum().sum())
        txt_hits = int((df[txt] == "NA").sum().sum())
        df[num] = df[num].mask(df[num] == -9999)
        df[txt] = df[txt].mask(df[txt] == "NA")
        return df, {"numeric_9999": num_hits, "text_NA": txt_hits}
