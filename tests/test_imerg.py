import json
from dataclasses import replace
from datetime import date, timedelta

import fakeredis
import numpy as np
import pytest
import xarray as xr

from nasa import Climate, OutsideNigeria
from nasa.cache import NullCache, RedisCache
from nasa.sources.imerg import EarthaccessFetcher, ImergError, read_point

ABUJA = (9.0765, 7.3986)
LATS = np.round(np.arange(3.95, 14.96, 0.1), 2)
LONS = np.round(np.arange(2.05, 15.06, 0.1), 2)


def make_day(path, day, dims=("time", "lon", "lat"), fill_cell=None, count=True, name="precipitation", days=1):
    lat_grid, lon_grid = np.meshgrid(LATS, LONS)
    base = (lat_grid + lon_grid / 100).astype("float32")
    if fill_cell is not None:
        base[fill_cell] = -9999.9
    arr = np.repeat(base[None], days, axis=0)
    if dims[1] == "lat":
        arr = arr.transpose(0, 2, 1)
    attrs = {"units": "mm/day"}
    data = {name: (dims, arr, attrs)}
    if count:
        data["precipitation_cnt"] = (dims, np.full(arr.shape, 48, dtype="float32"), {})
    times = np.array([np.datetime64(day)] * days) if days == 1 else np.array(
        [np.datetime64(day) + np.timedelta64(i, "D") for i in range(days)])
    ds = xr.Dataset(data, coords={"time": times, "lon": LONS, "lat": LATS})
    enc = {name: {"_FillValue": np.float32(-9999.9)}}
    ds.to_netcdf(path, engine="h5netcdf", encoding=enc)
    return path


def test_read_point_picks_the_nearest_cell_and_reports_it(tmp_path):
    f = make_day(tmp_path / "a.nc4", "2020-03-05")
    got = read_point(f, *ABUJA)
    assert got["date"] == "2020-03-05" and got["unit"] == "mm/day"
    assert got["cell_lat"] == pytest.approx(9.05) and got["cell_lon"] == pytest.approx(7.35)
    assert got["precipitation"] == pytest.approx(9.05 + 7.35 / 100, abs=1e-4)
    assert got["precipitation_cnt"] == 48


def test_read_point_does_not_depend_on_dimension_order(tmp_path):
    one = read_point(make_day(tmp_path / "a.nc4", "2020-03-05"), *ABUJA)
    two = read_point(make_day(tmp_path / "b.nc4", "2020-03-05", dims=("time", "lat", "lon")), *ABUJA)
    assert one["precipitation"] == pytest.approx(two["precipitation"])


def test_read_point_turns_the_fill_value_into_nan(tmp_path):
    i, j = int(np.argmin(abs(LONS - 7.35))), int(np.argmin(abs(LATS - 9.05)))
    f = make_day(tmp_path / "a.nc4", "2020-03-05", fill_cell=(i, j))
    got = read_point(f, *ABUJA)
    assert np.isnan(got["precipitation"]) and got["precipitation_cnt"] == 48


def test_read_point_without_count_variable(tmp_path):
    got = read_point(make_day(tmp_path / "a.nc4", "2020-03-05", count=False), *ABUJA)
    assert np.isnan(got["precipitation_cnt"]) and not np.isnan(got["precipitation"])


def test_read_point_refuses_a_point_far_from_the_grid(tmp_path):
    f = make_day(tmp_path / "a.nc4", "2020-03-05")
    with pytest.raises(ImergError, match="too far"):
        read_point(f, 40.0, 7.4)


def test_read_point_refuses_a_file_without_precipitation(tmp_path):
    f = make_day(tmp_path / "a.nc4", "2020-03-05", name="other")
    with pytest.raises(ImergError, match="no precipitation"):
        read_point(f, *ABUJA)


def test_read_point_refuses_a_multi_day_file(tmp_path):
    f = make_day(tmp_path / "a.nc4", "2020-03-05", days=2)
    with pytest.raises(ImergError, match="one day"):
        read_point(f, *ABUJA)


def value_for(day):
    return round(int(day.strftime("%Y%m%d")) % 53 / 10, 1)


class FakeFetcher:
    short_name = "GPM_3IMERGDF"
    version = "07"

    def __init__(self):
        self.calls = []
        self.fail_months = {}
        self.explode_on = None
        self.skip_days = set()
        self.offset = 0.0
        self.stray = False

    def month(self, first, last, lat, lon):
        self.calls.append((first, last))
        if self.explode_on == first.strftime("%Y-%m"):
            raise ImergError("login failed")
        rows = []
        for i in range((last - first).days + 1):
            d = first + timedelta(days=i)
            if d in self.skip_days:
                continue
            rows.append({"date": d.isoformat(), "cell_lat": 9.05, "cell_lon": 7.35,
                         "precipitation": value_for(d) + self.offset, "precipitation_cnt": 48.0, "unit": "mm/day"})
        if self.stray:
            nxt = last + timedelta(days=1)
            rows.append({"date": nxt.isoformat(), "cell_lat": 9.05, "cell_lon": 7.35,
                         "precipitation": 999.0, "precipitation_cnt": 48.0, "unit": "mm/day"})
        failed = [self.fail_months[first.strftime("%Y-%m")]] if first.strftime("%Y-%m") in self.fail_months else []
        return rows, failed


@pytest.fixture
def fetcher():
    return FakeFetcher()


@pytest.fixture
def clim(settings, tmp_path, fetcher):
    cfg = replace(settings, database_url=f"sqlite:///{(tmp_path / 'test.db').as_posix()}")
    c = Climate(cfg, cache=NullCache(), imerg_fetcher=fetcher)
    c.imerg.progress = False
    return c


def test_series_shape_columns_and_location(clim):
    df = clim.imerg.series(*ABUJA, "2020-01-01", "2020-03-31")
    assert len(df) == 91 and df["date"].is_unique and df["date"].is_monotonic_increasing
    assert list(df.columns) == ["date", "lat", "lon", "state", "geopolitical_zone", "cell_lat", "cell_lon",
                                "precipitation", "precipitation_cnt"]
    assert (df["state"] == "Federal Capital Territory").all()
    assert df["cell_lat"].iloc[0] == 9.05 and df["cell_lon"].iloc[0] == 7.35
    assert df.attrs["unit"] == "mm/day" and df.attrs["missing_days"] == 0 and df.attrs["provisional"] is False


def test_values_are_what_the_fetcher_returned(clim):
    df = clim.imerg.series(*ABUJA, "2020-02-10", "2020-02-14")
    want = [value_for(date(2020, 2, 10) + timedelta(days=i)) for i in range(5)]
    assert np.allclose(df["precipitation"].to_numpy(), want) and (df["precipitation_cnt"] == 48).all()
    rows = clim.store.read_where("imerg_daily")
    assert rows["precipitation"].dtype.kind == "f"


def test_second_call_and_a_new_object_read_the_database(clim, fetcher, settings, tmp_path):
    first = clim.imerg.series(*ABUJA, "2020-01-01", "2020-06-30")
    assert len(fetcher.calls) == 6
    again = clim.imerg.series(*ABUJA, "2020-01-01", "2020-06-30")
    assert len(fetcher.calls) == 6 and np.allclose(first["precipitation"], again["precipitation"])
    assert again["cell_lat"].iloc[0] == 9.05 and again.attrs["unit"] == "mm/day"
    cfg = replace(settings, database_url=f"sqlite:///{(tmp_path / 'test.db').as_posix()}")
    other = Climate(cfg, cache=NullCache(), imerg_fetcher=fetcher)
    other.imerg.progress = False
    sub = other.imerg.series(*ABUJA, "2020-02-10", "2020-03-05")
    assert len(sub) == 25 and len(fetcher.calls) == 6


def test_data_is_stored_a_whole_month_at_a_time(clim, fetcher):
    clim.imerg.series(*ABUJA, "2020-01-15", "2020-02-10")
    assert [(a, b) for a, b in fetcher.calls] == [(date(2020, 1, 1), date(2020, 1, 31)), (date(2020, 2, 1), date(2020, 2, 29))]
    assert len(clim.store.read_where("imerg_daily")) == 31 + 29
    assert sorted(clim.imerg.coverage()["month"]) == ["2020-01", "2020-02"]


def test_only_new_months_are_fetched(clim, fetcher):
    clim.imerg.series(*ABUJA, "2020-01-01", "2020-02-29")
    fetcher.calls.clear()
    df = clim.imerg.series(*ABUJA, "2020-01-01", "2020-04-30")
    assert [c[0] for c in fetcher.calls] == [date(2020, 3, 1), date(2020, 4, 1)] and len(df) == 121


def test_default_range_is_300_months(clim, fetcher):
    df = clim.imerg.series(*ABUJA)
    assert len(df) == 9131 and len(fetcher.calls) == 300
    assert fetcher.calls[0] == (date(2001, 1, 1), date(2001, 1, 31)) and fetcher.calls[-1][1] == date(2025, 12, 31)


def test_a_stopped_run_resumes_where_it_left_off(clim, fetcher):
    fetcher.explode_on = "2020-04"
    with pytest.raises(ImergError):
        clim.imerg.series(*ABUJA, "2020-01-01", "2020-06-30")
    assert sorted(clim.imerg.coverage()["month"]) == ["2020-01", "2020-02", "2020-03"]
    fetcher.explode_on = None
    fetcher.calls.clear()
    df = clim.imerg.series(*ABUJA, "2020-01-01", "2020-06-30")
    assert [c[0].month for c in fetcher.calls] == [4, 5, 6] and df["precipitation"].notna().all()


def test_failed_files_are_reported_and_retried_not_stored(clim, fetcher):
    fetcher.fail_months = {"2020-02": "OSError: boom"}
    df = clim.imerg.series(*ABUJA, "2020-01-01", "2020-03-31")
    assert df.attrs["failed"] == ["2020-02: OSError: boom"]
    assert sorted(clim.imerg.coverage()["month"]) == ["2020-01", "2020-03"]
    assert df["precipitation"].notna().all()
    fetcher.fail_months = {}
    fetcher.calls.clear()
    clim.imerg.series(*ABUJA, "2020-01-01", "2020-03-31")
    assert [c[0].month for c in fetcher.calls] == [2]


def test_days_without_a_file_show_as_missing(clim, fetcher):
    fetcher.skip_days = {date(2020, 1, 10), date(2020, 1, 11)}
    df = clim.imerg.series(*ABUJA, "2020-01-01", "2020-01-31")
    assert df.attrs["missing_days"] == 2 and df["precipitation"].isna().sum() == 2 and len(df) == 31


def test_recent_months_are_returned_but_not_stored(clim, fetcher):
    end = date.today() - timedelta(days=5)
    df = clim.imerg.series(*ABUJA, end - timedelta(days=3), end)
    assert df.attrs["provisional"] is True and len(df) == 4 and df["precipitation"].notna().all()
    assert clim.imerg.coverage().empty and clim.store.read_where("imerg_daily").empty
    n = len(fetcher.calls)
    clim.imerg.series(*ABUJA, end - timedelta(days=3), end)
    assert len(fetcher.calls) > n


def test_the_fetch_never_asks_past_today(clim, fetcher):
    clim.imerg.series(*ABUJA, date.today() - timedelta(days=2), date.today())
    assert all(last <= date.today() for _, last in fetcher.calls)


def test_refresh_downloads_again(clim, fetcher):
    clim.imerg.series(*ABUJA, "2020-01-01", "2020-01-31")
    clim.imerg.series(*ABUJA, "2020-01-01", "2020-01-31", refresh=True)
    assert len(fetcher.calls) == 2
    assert len(clim.store.read_where("imerg_daily")) == 31


def test_redis_serves_a_fresh_database(settings, tmp_path, fetcher):
    cache = RedisCache(fakeredis.FakeRedis(decode_responses=True), ttl=1000)

    def build(name):
        cfg = replace(settings, database_url=f"sqlite:///{(tmp_path / name).as_posix()}")
        c = Climate(cfg, cache=cache, imerg_fetcher=fetcher)
        c.imerg.progress = False
        return c

    build("a.db").imerg.series(*ABUJA, "2020-01-01", "2020-02-29")
    assert len(fetcher.calls) == 2
    df = build("b.db").imerg.series(*ABUJA, "2020-01-01", "2020-02-29")
    assert len(fetcher.calls) == 2 and df["precipitation"].notna().all() and df.attrs["unit"] == "mm/day"


def test_redis_does_not_keep_failed_or_recent_months(settings, tmp_path, fetcher):
    client = fakeredis.FakeRedis(decode_responses=True)
    cfg = replace(settings, database_url=f"sqlite:///{(tmp_path / 'a.db').as_posix()}")
    c = Climate(cfg, cache=RedisCache(client, ttl=1000), imerg_fetcher=fetcher)
    c.imerg.progress = False
    fetcher.fail_months = {"2020-01": "boom"}
    c.imerg.series(*ABUJA, "2020-01-01", "2020-01-31")
    c.imerg.series(*ABUJA, date.today() - timedelta(days=3), date.today())
    assert client.keys("*") == []


def test_points_outside_nigeria_and_bad_dates_are_refused(clim, fetcher):
    with pytest.raises(OutsideNigeria):
        clim.imerg.series(51.5, -0.12, "2020-01-01", "2020-01-31")
    for start, end in [("2020-02-01", "2020-01-01"), ("1990-01-01", "2000-01-01"), ("2020-01-01", "2999-01-01")]:
        with pytest.raises(ValueError):
            clim.imerg.series(*ABUJA, start, end)
    assert fetcher.calls == []


class FakeEarthaccess:
    def __init__(self, folder):
        self.folder = folder
        self.logins = []
        self.searches = []
        self.good_login = True
        self.opened = []

    def login(self, strategy="all"):
        self.logins.append(strategy)

        class Auth:
            authenticated = self.good_login and strategy == "environment"

        return Auth()

    def search_data(self, **kw):
        self.searches.append(kw)
        first = date.fromisoformat(kw["temporal"][0])
        last = date.fromisoformat(kw["temporal"][1])
        days = [first + timedelta(days=i) for i in range((last - first).days + 1)]
        return [d for d in days if d.day != 3]

    def open(self, granules):
        files = []
        for d in granules:
            path = self.folder / f"{d}.nc4"
            if d.day == 5:
                path.write_bytes(b"not a netcdf file")
            else:
                make_day(path, d.isoformat())
            f = open(path, "rb")
            self.opened.append(f)
            files.append(f)
        return files


def test_earthaccess_fetcher_reads_real_files_and_collects_bad_ones(tmp_path):
    ea = FakeEarthaccess(tmp_path)
    fetcher = EarthaccessFetcher(earthaccess=ea)
    rows, failed = fetcher.month(date(2020, 1, 1), date(2020, 1, 10), *ABUJA)
    assert len(rows) == 8 and len(failed) == 1
    assert ea.searches[0] == {"short_name": "GPM_3IMERGDF", "version": "07", "temporal": ("2020-01-01", "2020-01-10")}
    assert ea.logins == ["environment"]
    assert all(f.closed for f in ea.opened)
    assert rows[0]["date"] == "2020-01-01" and rows[0]["cell_lon"] == pytest.approx(7.35)


def test_earthaccess_fetcher_logs_in_once_and_reads_env_for_the_product(tmp_path, monkeypatch):
    monkeypatch.setenv("IMERG_SHORT_NAME", "GPM_3IMERGDL")
    monkeypatch.setenv("IMERG_VERSION", "08")
    ea = FakeEarthaccess(tmp_path)
    fetcher = EarthaccessFetcher(earthaccess=ea)
    fetcher.month(date(2020, 1, 1), date(2020, 1, 2), *ABUJA)
    fetcher.month(date(2020, 2, 1), date(2020, 2, 2), *ABUJA)
    assert ea.logins == ["environment"]
    assert ea.searches[0]["short_name"] == "GPM_3IMERGDL" and ea.searches[0]["version"] == "08"


def test_earthaccess_fetcher_says_what_to_fix_when_login_fails(tmp_path):
    ea = FakeEarthaccess(tmp_path)
    ea.good_login = False
    with pytest.raises(ImergError, match="EARTHDATA_USERNAME"):
        EarthaccessFetcher(earthaccess=ea).month(date(2020, 1, 1), date(2020, 1, 2), *ABUJA)
    assert ea.logins == ["environment", "netrc"]


def test_whole_chain_with_real_files(settings, tmp_path):
    ea = FakeEarthaccess(tmp_path)
    cfg = replace(settings, database_url=f"sqlite:///{(tmp_path / 'chain.db').as_posix()}")
    c = Climate(cfg, cache=NullCache(), imerg_fetcher=EarthaccessFetcher(earthaccess=ea))
    c.imerg.progress = False
    df = c.imerg.series(*ABUJA, "2020-01-01", "2020-01-31")
    assert len(df) == 31 and df.attrs["missing_days"] == 2 and len(df.attrs["failed"]) == 1
    assert np.isnan(df.loc[df["date"] == "2020-01-03", "precipitation"]).all()
    assert df.loc[df["date"] == "2020-01-02", "precipitation"].iloc[0] == pytest.approx(9.05 + 0.0735, abs=1e-4)
    assert json.dumps(df.attrs["unit"]) == '"mm/day"'


def test_files_from_outside_the_month_are_dropped(clim, fetcher):
    fetcher.stray = True
    df = clim.imerg.series(*ABUJA, "2020-01-01", "2020-02-29")
    assert len(clim.store.read_where("imerg_daily")) == 60
    assert (df["precipitation"] < 100).all() and df["precipitation"].notna().all()


def test_refresh_with_a_failure_keeps_old_days_and_takes_new_ones(clim, fetcher):
    clim.imerg.series(*ABUJA, "2020-01-01", "2020-01-31")
    fetcher.offset = 100.0
    fetcher.skip_days = {date(2020, 1, 10)}
    fetcher.fail_months = {"2020-01": "boom"}
    df = clim.imerg.series(*ABUJA, "2020-01-01", "2020-01-31", refresh=True)
    by_day = df.set_index("date")["precipitation"]
    assert by_day["2020-01-09"] == pytest.approx(value_for(date(2020, 1, 9)) + 100)
    assert by_day["2020-01-10"] == pytest.approx(value_for(date(2020, 1, 10)))
    assert len(df.attrs["failed"]) == 1
    assert len(clim.store.read_where("imerg_daily")) == 31
