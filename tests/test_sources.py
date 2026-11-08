import numpy as np
import pandas as pd
import pytest
from shapely.geometry import Point

from nasa import SoilExplorer
from nasa.cache import NullCache

from .conftest import needs_data

pytestmark = needs_data


@pytest.fixture(scope="module")
def explorer(settings):
    return SoilExplorer(settings, cache=NullCache())


def independent_states(regions, lats, lons):
    out = []
    for lat, lon in zip(lats, lons):
        pt = Point(lon, lat)
        hit = [s for s, g in zip(regions.states["state"], regions.states.geometry) if g.covers(pt)]
        out.append(hit[0] if hit else None)
    return out


def test_afsp_shape_and_ids(explorer):
    df = explorer.afsp.frame()
    assert len(df) == 5894
    note = explorer.afsp.note
    assert note["profiles"] == 1251 and note["profiles_without_layers"] == 1
    assert df["profileid"].nunique() == note["profiles"] - note["profiles_without_layers"]
    assert df["layerid"].is_unique
    assert df.columns.is_unique
    assert df["profileid"].str.startswith("NG").all()


def test_afsp_missing_markers_are_gone(explorer):
    df = explorer.afsp.frame()
    num = df.select_dtypes("number")
    assert not (num == -9999).any().any()
    txt = df.select_dtypes(exclude="number")
    assert not (txt == "NA").any().any()
    assert explorer.afsp.note["missing_markers_set_to_nan"]["numeric_9999"] > 0


def test_afsp_coordinates_and_states(explorer):
    df = explorer.afsp.frame()
    assert not ((df["lat"] == 0) & (df["lon"] == 0)).any()
    no_coords = df["lat"].isna()
    assert no_coords.sum() == explorer.afsp.note["layers_without_coordinates"]
    assert df.loc[~no_coords, "state"].notna().all()
    assert df.loc[no_coords, "state"].isna().all()
    assert df["lat"].dropna().between(4, 14).all() and df["lon"].dropna().between(2, 15).all()


def test_afsp_states_match_independent_check(explorer, regions):
    df = explorer.afsp.frame().dropna(subset=["lat"])
    inside = df[df["state_distance_km"] == 0].sample(300, random_state=1)
    assert inside["state"].tolist() == independent_states(regions, inside["lat"], inside["lon"])


def test_afsp_values_survive_the_cleaning(explorer):
    df = explorer.afsp.frame()
    assert 3 < df["phh2o"].min() and df["phh2o"].max() < 11
    assert df["orgc"].min() >= 0
    assert df["t_year"].dropna().between(1900, 2026).all()


def test_samples_only_inside_nigeria(explorer, regions):
    df = explorer.samples.frame()
    note = explorer.samples.note
    assert len(df) == note["rows_inside_nigeria"] and note["rows_in_file"] == 49225
    assert df["state"].notna().all() and (df["state_distance_km"] == 0).all()
    assert {"lat", "lon", "ph", "carbon_organic", "horizon_upper", "horizon_lower"} <= set(df.columns)
    sample = df.sample(300, random_state=2)
    assert sample["state"].tolist() == independent_states(regions, sample["lat"], sample["lon"])


def test_samples_match_raw_rows_one_to_one(explorer, regions):
    raw = pd.read_csv(explorer.settings.isda_csv, encoding="utf-8-sig")
    box = raw[raw["longitude"].between(2.0, 15.5) & raw["latitude"].between(3.5, 14.5)]
    states = independent_states(regions, box["latitude"], box["longitude"])
    want = box[[s is not None for s in states]].reset_index(drop=True)
    want = want.rename(columns={"longitude": "lon", "latitude": "lat"})
    got = explorer.samples.frame()
    assert len(want) == len(got)
    for col in want.columns:
        if pd.api.types.is_numeric_dtype(want[col]):
            same = (want[col].isna() & got[col].isna()) | np.isclose(want[col], got[col])
        else:
            same = (want[col].isna() & got[col].isna()) | (want[col].astype(str) == got[col].astype(str))
        assert same.all(), col


def test_afsp_matches_raw_dbf(explorer):
    import pyogrio

    path = explorer.settings.afsp_dir / "AfSP012Qry_Layers.dbf"
    raw = pyogrio.read_dataframe(path, where="ProfileID LIKE 'NG%'", read_geometry=False)
    raw.columns = [c.lower() for c in raw.columns]
    raw = raw.set_index("layerid")
    got = explorer.afsp.frame().set_index("layerid").reindex(raw.index)
    assert got["profileid"].notna().all()
    for col in raw.select_dtypes("number").columns:
        want = raw[col].replace(-9999, np.nan)
        same = (want.isna() & got[col].isna()) | np.isclose(want, got[col])
        assert same.all(), col


def test_soil_map_attributes_match_raw_shapefile(explorer):
    import geopandas as gpd

    raw = gpd.read_file(explorer.soil_map._shapefile())
    gdf = explorer.soil_map.frame()
    rows = gdf["polygon_row"].to_numpy()
    pairs = {"MAPPING_UN": "mapping_unit", "DRAINAGE": "drainage", "Depth": "depth_class",
             "SOIL_TEXTU": "texture", "SOIL_PH": "ph_range", "MAJOR_CROP": "major_crops"}
    for src, dst in pairs.items():
        assert (raw[src].astype(str).str.strip().to_numpy()[rows] == gdf[dst].astype(str).to_numpy()).all(), dst


def test_samples_metadata_loads(explorer):
    meta = explorer.samples.metadata()
    assert {"column_name", "description", "unit"} <= set(meta.columns)


def test_soil_map_pieces(explorer, settings):
    gdf = explorer.soil_map.frame()
    note = explorer.soil_map.note
    assert len(gdf) == note["pieces_after_state_split"]
    assert gdf["state"].notna().all() and gdf["geopolitical_zone"].notna().all()
    assert gdf.geometry.notna().all() and (~gdf.geometry.is_empty).all()
    share = gdf.groupby("polygon_row")["piece_share"].sum()
    assert share.max() <= 1.001
    assert note["duplicate_source_ids"] == [389]
    assert gdf["polygon_row"].nunique() + len(note["polygon_rows_with_no_state"]) == note["polygons_in_file"]


@pytest.mark.parametrize("name,lat,lon,unit,state", [
    ("abuja", 9.0765, 7.3986, "22c", "Federal Capital Territory"),
    ("kano", 11.9964, 8.5167, "2a", "Kano"),
    ("port_harcourt", 4.8156, 7.0498, "1b", "Rivers"),
    ("maiduguri", 11.8333, 13.15, "15a", "Borno"),
])
def test_soil_map_point_lookup_matches_direct_shapefile_read(explorer, name, lat, lon, unit, state):
    hit = explorer.soil_map.at_point(lat, lon)
    assert len(hit) == 1
    row = hit.iloc[0]
    assert row["match"] == "inside" and row["distance_km"] == 0
    assert row["mapping_unit"] == unit and row["state"] == state


def test_soil_map_lagos_uses_nearest_and_says_so(explorer):
    hit = explorer.soil_map.at_point(6.45, 3.39)
    assert hit.iloc[0]["match"] == "nearest" and hit.iloc[0]["distance_km"] > 0


def test_point_sources_near(explorer):
    near = explorer.afsp.near(9.0765, 7.3986, radius_km=25)
    assert (near["distance_km"] <= 25).all() and near["distance_km"].is_monotonic_increasing
    assert len(near) > 0
    assert explorer.afsp.nearest_km(9.0765, 7.3986) == pytest.approx(near["distance_km"].min(), abs=0.05)
    assert explorer.samples.nearest_km(9.0765, 7.3986) > 25


def test_cached_tables_equal_fresh_builds(explorer):
    for src in (explorer.afsp, explorer.samples):
        cached = src.frame()
        fresh = src.frame(refresh=True)
        pd.testing.assert_frame_equal(cached, fresh, check_dtype=False)
    cached = explorer.soil_map.frame()
    fresh = explorer.soil_map.frame(refresh=True)
    pd.testing.assert_frame_equal(pd.DataFrame(cached.drop(columns="geometry")),
                                  pd.DataFrame(fresh.drop(columns="geometry")), check_dtype=False)
    assert np.all(cached.geometry.geom_equals_exact(fresh.geometry, tolerance=1e-6))


def test_second_explorer_reads_from_database(settings, monkeypatch):
    SoilExplorer(settings, cache=NullCache()).tables()
    second = SoilExplorer(settings, cache=NullCache())

    def boom(*a, **k):
        raise AssertionError("rebuilt instead of reading the database")

    for src in (second.afsp, second.samples, second.soil_map):
        monkeypatch.setattr(src, "build", boom)
    tables = second.tables()
    assert len(tables["afsp"]) == 5894
    assert second.afsp.note["layers"] == 5894


def test_overview(explorer):
    ov = explorer.overview()
    assert list(ov.index) == ["afsp", "isda_samples", "soil_map", "isdasoil_api"]
    assert ov.loc["afsp", "states"] == 37


def test_at_point_with_fake_api(settings, session):
    ex = SoilExplorer(settings, session=session, cache=NullCache())
    out = ex.at_point(9.0765, 7.3986, radius_km=25)
    assert out["location"]["state"] == "Federal Capital Territory"
    assert out["location"]["geopolitical_zone"] == "North Central"
    assert out["location"]["nearest_afsp_km"] < 25 and out["location"]["nearest_isda_sample_km"] > 25
    assert set(out) == {"location", "afsp", "isda_samples", "soil_map", "isdasoil"}
    assert out["soil_map"].iloc[0]["drainage"] == "Poorly Drained"
    assert len(out["isdasoil"]) > 0
    assert "isdasoil" not in ex.at_point(9.0765, 7.3986, include_api=False)
