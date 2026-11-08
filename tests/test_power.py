import json
from dataclasses import replace
from datetime import date, timedelta

import fakeredis
import numpy as np
import pytest

from nasa import Climate, OutsideNigeria
from nasa.cache import NullCache, RedisCache
from nasa.sources.nasa_power import DEFAULT_PARAMETERS, NasaPowerError

from .conftest import FakeResponse

UNITS = {"T2M": "C", "PRECTOTCORR": "mm/day", "GWETROOT": "1", "T2M_MAX": "C"}


def value_for(name, day):
    return round((int(day.strftime("%Y%m%d")) % 97) / 10 + len(name), 2)


class FakePower:
    def __init__(self):
        self.gets = []
        self.status = []
        self.hole = {"T2M": "20050101"}

    def get(self, url, params=None, timeout=None):
        self.gets.append(params)
        if self.status:
            return FakeResponse(self.status.pop(0), "temporary")
        names = params["parameters"].split(",")
        if "BOGUS" in names:
            return FakeResponse(422, {"messages": ["BOGUS is not a valid parameter"]})
        first = date(int(params["start"][:4]), int(params["start"][4:6]), int(params["start"][6:]))
        last = date(int(params["end"][:4]), int(params["end"][4:6]), int(params["end"][6:]))
        days = [first + timedelta(days=i) for i in range((last - first).days + 1)]
        body = {n: {d.strftime("%Y%m%d"): value_for(n, d) for d in days} for n in names}
        for n, stamp in self.hole.items():
            if n in body and stamp in body[n]:
                body[n][stamp] = -999.0
        return FakeResponse(200, {
            "type": "Feature",
            "geometry": {"type": "Point", "coordinates": [7.5, 9.0, 480.5]},
            "properties": {"parameter": body},
            "header": {"fill_value": -999.0},
            "messages": [],
            "parameters": {n: {"units": UNITS.get(n, "x"), "longname": n} for n in names},
        })


@pytest.fixture
def power_session():
    return FakePower()


@pytest.fixture
def climate(settings, tmp_path, power_session):
    cfg = replace(settings, database_url=f"sqlite:///{(tmp_path / 'test.db').as_posix()}")
    return Climate(cfg, session=power_session, cache=NullCache())


ABUJA = (9.0765, 7.3986)


def test_series_shape_columns_and_location(climate):
    df = climate.power.series(*ABUJA, "2020-01-01", "2020-12-31", ["T2M", "PRECTOTCORR"])
    assert len(df) == 366 and df["date"].is_unique and df["date"].is_monotonic_increasing
    assert list(df.columns) == ["date", "lat", "lon", "state", "geopolitical_zone", "grid_lat", "grid_lon", "T2M", "PRECTOTCORR"]
    assert (df["state"] == "Federal Capital Territory").all() and (df["geopolitical_zone"] == "North Central").all()
    assert df["grid_lat"].iloc[0] == 9.0 and df["grid_lon"].iloc[0] == 7.5
    assert df.attrs["units"] == {"T2M": "C", "PRECTOTCORR": "mm/day"}
    assert df.attrs["elevation_m"] == 480.5 and df.attrs["provisional"] is False


def test_values_match_what_the_api_sent(climate):
    df = climate.power.series(*ABUJA, "2020-03-01", "2020-03-10", ["T2M"])
    want = [value_for("T2M", date(2020, 3, 1) + timedelta(days=i)) for i in range(10)]
    assert np.allclose(df["T2M"].to_numpy(), want)


def test_fill_value_becomes_nan_and_is_counted(climate):
    df = climate.power.series(*ABUJA, "2005-01-01", "2005-01-05", ["T2M", "PRECTOTCORR"])
    assert np.isnan(df.loc[0, "T2M"]) and df["T2M"].iloc[1:].notna().all()
    assert df.attrs["missing_days"] == {"T2M": 1, "PRECTOTCORR": 0}


def test_second_call_comes_from_the_database(climate, power_session):
    first = climate.power.series(*ABUJA, "2020-01-01", "2020-06-30", ["T2M", "PRECTOTCORR"])
    calls = len(power_session.gets)
    again = climate.power.series(*ABUJA, "2020-01-01", "2020-06-30", ["T2M", "PRECTOTCORR"])
    assert len(power_session.gets) == calls
    assert np.allclose(first["T2M"], again["T2M"]) and again.attrs["units"] == first.attrs["units"]
    assert again["grid_lat"].iloc[0] == 9.0 and again.attrs["elevation_m"] == 480.5


def test_a_narrower_range_and_a_second_climate_object_reuse_the_database(climate, power_session, settings, tmp_path):
    climate.power.series(*ABUJA, "2020-01-01", "2020-12-31", ["T2M"])
    calls = len(power_session.gets)
    cfg = replace(settings, database_url=f"sqlite:///{(tmp_path / 'test.db').as_posix()}")
    other = Climate(cfg, session=power_session, cache=NullCache())
    sub = other.power.series(*ABUJA, "2020-02-01", "2020-02-29", ["T2M"])
    assert len(sub) == 29 and len(power_session.gets) == calls


def test_only_missing_parameters_are_fetched(climate, power_session):
    climate.power.series(*ABUJA, "2020-01-01", "2020-01-31", ["T2M"])
    power_session.gets.clear()
    df = climate.power.series(*ABUJA, "2020-01-01", "2020-01-31", ["T2M", "T2M_MAX"])
    assert len(power_session.gets) == 1 and power_session.gets[0]["parameters"] == "T2M_MAX"
    assert df["T2M_MAX"].notna().all() and df["T2M"].notna().all()


def test_wider_range_is_fetched_again_and_replaces_rows(climate, power_session):
    climate.power.series(*ABUJA, "2020-03-01", "2020-03-31", ["T2M"])
    power_session.gets.clear()
    df = climate.power.series(*ABUJA, "2020-01-01", "2020-12-31", ["T2M"])
    assert len(power_session.gets) == 1 and len(df) == 366
    rows = climate.power.store.read_where("power_daily", parameter="T2M")
    assert len(rows) == 366 and rows["date"].is_unique


def test_default_range_is_split_in_five_year_requests(climate, power_session):
    df = climate.power.series(*ABUJA, parameters=["T2M"])
    assert len(df) == 9131
    spans = [(g["start"], g["end"]) for g in power_session.gets]
    assert spans == [("20010101", "20051231"), ("20060101", "20101231"), ("20110101", "20151231"),
                     ("20160101", "20201231"), ("20210101", "20251231")]


def test_default_parameters_all_requested_in_one_call_per_chunk(climate, power_session):
    climate.power.series(*ABUJA, "2020-01-01", "2020-01-10")
    assert len(power_session.gets) == 1
    assert power_session.gets[0]["parameters"].split(",") == DEFAULT_PARAMETERS
    assert power_session.gets[0]["community"] == "AG" and power_session.gets[0]["format"] == "JSON"


def test_recent_days_are_not_stored(climate, power_session):
    end = date.today() - timedelta(days=10)
    df = climate.power.series(*ABUJA, end - timedelta(days=5), end, ["T2M"])
    assert df.attrs["provisional"] is True and len(df) == 6
    assert climate.power.coverage().empty
    calls = len(power_session.gets)
    climate.power.series(*ABUJA, end - timedelta(days=5), end, ["T2M"])
    assert len(power_session.gets) == calls + 1


def test_refresh_downloads_again(climate, power_session):
    climate.power.series(*ABUJA, "2020-01-01", "2020-01-10", ["T2M"])
    calls = len(power_session.gets)
    climate.power.series(*ABUJA, "2020-01-01", "2020-01-10", ["T2M"], refresh=True)
    assert len(power_session.gets) == calls + 1


def test_redis_serves_a_fresh_database(settings, tmp_path, power_session):
    cache = RedisCache(fakeredis.FakeRedis(decode_responses=True), ttl=1000)
    one = Climate(replace(settings, database_url=f"sqlite:///{(tmp_path / 'a.db').as_posix()}"), session=power_session, cache=cache)
    one.power.series(*ABUJA, "2020-01-01", "2020-01-31", ["T2M"])
    calls = len(power_session.gets)
    two = Climate(replace(settings, database_url=f"sqlite:///{(tmp_path / 'b.db').as_posix()}"), session=power_session, cache=cache)
    df = two.power.series(*ABUJA, "2020-01-01", "2020-01-31", ["T2M"])
    assert len(power_session.gets) == calls and df["T2M"].notna().all()


def test_retries_a_busy_server(climate, power_session, monkeypatch):
    monkeypatch.setattr("time.sleep", lambda s: None)
    power_session.status = [503, 429]
    df = climate.power.series(*ABUJA, "2020-01-01", "2020-01-05", ["T2M"])
    assert len(df) == 5 and len(power_session.gets) == 3


def test_bad_parameter_error_carries_the_api_message(climate):
    with pytest.raises(NasaPowerError, match="422.*BOGUS"):
        climate.power.series(*ABUJA, "2020-01-01", "2020-01-05", ["BOGUS"])
    assert climate.power.coverage().empty


def test_a_missing_parameter_in_the_answer_is_an_error(climate, power_session):
    original = power_session.get

    def drop(url, params=None, timeout=None):
        resp = original(url, params, timeout)
        js = resp.json()
        js["properties"]["parameter"].pop("T2M")
        return FakeResponse(200, js)

    power_session.get = drop
    with pytest.raises(NasaPowerError, match="did not return T2M"):
        climate.power.series(*ABUJA, "2020-01-01", "2020-01-05", ["T2M"])


@pytest.mark.parametrize("lat,lon", [(51.5, -0.12), (0.0, 0.0)])
def test_points_outside_nigeria_are_refused_before_any_request(climate, power_session, lat, lon):
    with pytest.raises(OutsideNigeria):
        climate.power.series(lat, lon, "2020-01-01", "2020-01-05", ["T2M"])
    assert power_session.gets == []


@pytest.mark.parametrize("start,end", [("2020-02-01", "2020-01-01"), ("1970-01-01", "1980-01-01"), ("2020-01-01", "2999-01-01")])
def test_bad_dates_are_refused(climate, start, end):
    with pytest.raises(ValueError):
        climate.power.series(*ABUJA, start, end, ["T2M"])


@pytest.mark.parametrize("names", [[], ["T2M;DROP"], [f"P{i}" for i in range(21)]])
def test_bad_parameter_lists_are_refused(climate, names):
    with pytest.raises(ValueError):
        climate.power.series(*ABUJA, "2020-01-01", "2020-01-05", names)


def test_date_formats_and_parameter_case(climate, power_session):
    df = climate.power.series(*ABUJA, "20200101", date(2020, 1, 3), ["t2m", "T2M"])
    assert len(df) == 3 and "T2M" in df.columns and power_session.gets[0]["parameters"] == "T2M"
    assert json.dumps(power_session.gets[0]["start"]) == '"20200101"'


def test_a_later_end_date_is_fetched_again(climate, power_session):
    climate.power.series(*ABUJA, "2020-01-01", "2020-06-30", ["T2M"])
    power_session.gets.clear()
    df = climate.power.series(*ABUJA, "2020-01-01", "2020-12-31", ["T2M"])
    assert len(power_session.gets) == 1 and len(df) == 366 and df["T2M"].notna().all()
