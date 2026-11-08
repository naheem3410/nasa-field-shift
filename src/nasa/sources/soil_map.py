import geopandas as gpd
import pandas as pd
from shapely.geometry import Point

from ..geo import METRIC_CRS
from .base import FileSource

RENAME = {
    "ID": "polygon_id",
    "MAPPING_UN": "mapping_unit",
    "GEOLOGY": "geology",
    "SLOPE": "slope_range",
    "ECOLOGICAL": "ecological_zone",
    "DRAINAGE": "drainage",
    "SOIL_PH": "ph_range",
    "PH_DESCRIP": "ph_description",
    "SUITABILIT": "suitability",
    "SOIL_TEXTU": "texture",
    "SOIL_CLASS": "soil_class_usda",
    "SOIL_CLA_1": "soil_class_fao",
    "DISTRIBUTI": "distribution",
    "MAJOR_CROP": "major_crops",
    "Depth": "depth_class",
}


class NigeriaSoilMap(FileSource):
    table = "soil_map_nigeria"

    def _shapefile(self):
        found = sorted(self.settings.soil_map_dir.glob("*.shp"))
        if len(found) != 1:
            raise FileNotFoundError(f"expected one .shp in {self.settings.soil_map_dir}, found {len(found)}")
        return found[0]

    def inputs(self):
        shp = self._shapefile()
        return [shp, shp.with_suffix(".dbf")]

    def build(self):
        raw = gpd.read_file(self._shapefile())
        # the .prj says assumed NAD27 but the values are plain lon/lat, shift is tiny at this map scale
        raw = raw.set_crs(4326, allow_override=True).rename(columns=RENAME)
        raw.columns = [c.lower() for c in raw.columns]
        for col in raw.columns.drop("geometry"):
            if raw[col].dtype == object or str(raw[col].dtype) == "str":
                raw[col] = raw[col].str.strip()
        raw["geometry"] = raw.geometry.make_valid()
        # the source ids are not unique (id 389 appears twice), so keep the row number too
        raw["polygon_row"] = range(len(raw))

        pieces = self.regions.split_polygons(raw)
        reps = pieces.geometry.representative_point()
        pieces["rep_lat"] = reps.y
        pieces["rep_lon"] = reps.x
        front = ["polygon_row", "polygon_id", "mapping_unit", "state", "geopolitical_zone", "piece_km2", "piece_share"]
        pieces = pieces[front + [c for c in pieces.columns if c not in front]]
        lost = sorted(set(raw["polygon_row"]) - set(pieces["polygon_row"]))
        note = {
            "polygons_in_file": int(len(raw)),
            "pieces_after_state_split": int(len(pieces)),
            "polygon_rows_with_no_state": [int(i) for i in lost],
            "duplicate_source_ids": [int(i) for i in raw.loc[raw["polygon_id"].duplicated(keep=False), "polygon_id"].unique()],
        }
        return pieces, note

    def to_store(self, gdf):
        df = pd.DataFrame(gdf.drop(columns="geometry"))
        df["geometry_wkt"] = gdf.geometry.to_wkt()
        return df

    def from_store(self, df):
        geom = gpd.GeoSeries.from_wkt(df["geometry_wkt"], crs=4326)
        return gpd.GeoDataFrame(df.drop(columns="geometry_wkt"), geometry=geom, crs=4326)

    def at_point(self, lat, lon):
        gdf = self.frame()
        pt = Point(lon, lat)
        hit = gdf[gdf.covers(pt)]
        if len(hit):
            return hit.assign(match="inside", distance_km=0.0)
        metric = gdf.to_crs(METRIC_CRS)
        here = gpd.GeoSeries([pt], crs=4326).to_crs(METRIC_CRS).iloc[0]
        dist = metric.distance(here)
        return gdf.loc[[dist.idxmin()]].assign(match="nearest", distance_km=round(float(dist.min()) / 1000, 1))
