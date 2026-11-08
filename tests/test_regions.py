import pandas as pd
import pytest
from shapely.geometry import Point

from nasa.regions import GROUPS, ZONES, OutsideNigeria

CITIES = {
    "abuja": (9.0765, 7.3986, "Federal Capital Territory", "North Central"),
    "kano": (11.9964, 8.5167, "Kano", "North West"),
    "lagos": (6.5244, 3.3792, "Lagos", "South West"),
    "port_harcourt": (4.8156, 7.0498, "Rivers", "South South"),
    "maiduguri": (11.8333, 13.1500, "Borno", "North East"),
    "enugu": (6.4584, 7.5464, "Enugu", "South East"),
}


def test_has_37_states_and_known_zone_sizes(regions):
    assert len(regions.states) == 37
    assert regions.states["geopolitical_zone"].notna().all()
    sizes = {zone: len(states) for zone, states in GROUPS.items()}
    assert sizes == {"North Central": 7, "North East": 6, "North West": 7,
                     "South East": 5, "South South": 6, "South West": 6}
    assert len(ZONES) == 37


@pytest.mark.parametrize("name", list(CITIES))
def test_locate_cities(regions, name):
    lat, lon, state, zone = CITIES[name]
    assert regions.locate(lat, lon) == {"state": state, "geopolitical_zone": zone}


@pytest.mark.parametrize("lat,lon", [(-1.29, 36.82), (5.6, -0.2), (0.0, 0.0), (20.0, 8.0)])
def test_outside_nigeria_raises(regions, lat, lon):
    with pytest.raises(OutsideNigeria):
        regions.locate(lat, lon)


def test_tag_points_handles_bad_values(regions):
    df = pd.DataFrame({"lat": [9.0765, None, "x", 95.0, 11.9964], "lon": [7.3986, 8.0, 8.0, 8.0, 8.5167]})
    out = regions.tag_points(df)
    assert out["state"].tolist()[0] == "Federal Capital Territory"
    assert out["state"].tolist()[4] == "Kano"
    assert out["state"].iloc[1:4].isna().all()
    assert len(out) == 5


def test_tag_points_keeps_index_and_is_repeatable(regions):
    df = pd.DataFrame({"lat": [6.5244, 9.0765], "lon": [3.3792, 7.3986]}, index=[10, 20])
    once = regions.tag_points(df)
    twice = regions.tag_points(once)
    assert once.index.tolist() == [10, 20]
    pd.testing.assert_frame_equal(once, twice)


def test_tolerance_assigns_nearby_sea_point_only_when_allowed(regions):
    lagos = regions.states.set_index("state").geometry["Lagos"]
    south = lagos.bounds[1]
    df = pd.DataFrame({"lat": [south - 0.05], "lon": [3.4]})
    strict = regions.tag_points(df, max_km=0)
    loose = regions.tag_points(df, max_km=20)
    assert strict["state"].isna().all()
    assert loose["state"].iloc[0] == "Lagos"
    assert loose["state_distance_km"].iloc[0] > 0


def test_every_state_point_matches_independent_check(regions):
    rep = regions.states.geometry.representative_point()
    df = pd.DataFrame({"lat": rep.y, "lon": rep.x})
    out = regions.tag_points(df)
    assert out["state"].tolist() == regions.states["state"].tolist()
    for geom, lat, lon in zip(regions.states.geometry, rep.y, rep.x):
        assert geom.covers(Point(lon, lat))
