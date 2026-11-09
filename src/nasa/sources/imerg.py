import json
import logging
from datetime import date, datetime, timedelta, timezone

import numpy as np
import pandas as pd

from ..config import env_setting

log = logging.getLogger("nasa")

SHORT_NAME = "GPM_3IMERGDF"
VERSION = "07"
VARIABLE = "precipitation"
COUNT = "precipitation_cnt"
VALUES_TABLE = "imerg_daily"
COVERAGE_TABLE = "imerg_coverage"
KEY_VERSION = "1"
DEFAULT_START = "2001-01-01"
DEFAULT_END = "2025-12-31"
FIRST_DAY = date(1998, 1, 1)
# the final run reaches the archive about four months late, newer months are not stored
FINAL_AFTER_DAYS = 130
# a cell is 0.1 degrees wide, so the nearest centre can be at most half of that away
MAX_CELL_GAP = 0.06


class ImergError(RuntimeError):
    pass


def _day(value):
    if isinstance(value, datetime):
        return value.date()
    if isinstance(value, date):
        return value
    for fmt in ("%Y-%m-%d", "%Y%m%d"):
        try:
            return datetime.strptime(str(value).strip(), fmt).date()
        except ValueError:
            pass
    raise ValueError(f"cannot read the date {value!r}, use YYYY-MM-DD")


def _months(start, end):
    cur = date(start.year, start.month, 1)
    while cur <= end:
        nxt = date(cur.year + (cur.month == 12), cur.month % 12 + 1, 1)
        yield cur, nxt - timedelta(days=1)
        cur = nxt


def read_point(handle, lat, lon):
    import xarray as xr

    with xr.open_dataset(handle, engine="h5netcdf") as ds:
        if VARIABLE not in ds:
            raise ImergError(f"the file has no {VARIABLE} variable, it has {list(ds.data_vars)}")
        pick = ds.sel(lat=lat, lon=lon, method="nearest")
        cell_lat, cell_lon = float(pick["lat"]), float(pick["lon"])
        if abs(cell_lat - lat) > MAX_CELL_GAP or abs(cell_lon - lon) > MAX_CELL_GAP:
            raise ImergError(f"nearest cell ({cell_lat}, {cell_lon}) is too far from ({lat}, {lon})")
        value = np.asarray(pick[VARIABLE].values, dtype=float).reshape(-1)
        stamps = np.asarray(ds["time"].values).reshape(-1)
        if value.size != 1 or stamps.size != 1:
            raise ImergError("expected one day in each daily file")
        count = np.nan
        if COUNT in ds:
            count = float(np.asarray(pick[COUNT].values, dtype=float).reshape(-1)[0])
        return {
            "date": str(stamps[0])[:10], "cell_lat": cell_lat, "cell_lon": cell_lon,
            "precipitation": float(value[0]), "precipitation_cnt": count,
            "unit": ds[VARIABLE].attrs.get("units"),
        }


class EarthaccessFetcher:
    def __init__(self, short_name=None, version=None, earthaccess=None):
        self.short_name = short_name or env_setting("IMERG_SHORT_NAME", SHORT_NAME)
        self.version = version or env_setting("IMERG_VERSION", VERSION)
        self._ea = earthaccess
        self._ready = False

    def _login(self):
        if self._ea is None:
            import earthaccess

            self._ea = earthaccess
        if self._ready:
            return
        for strategy in ("environment", "netrc"):
            try:
                if self._ea.login(strategy=strategy).authenticated:
                    self._ready = True
                    return
            except Exception as err:
                log.debug("earthdata login by %s failed: %s", strategy, err)
        raise ImergError("Earthdata login failed, set EARTHDATA_USERNAME and EARTHDATA_PASSWORD in the .env file")

    def month(self, first, last, lat, lon):
        self._login()
        granules = self._ea.search_data(
            short_name=self.short_name, version=self.version,
            temporal=(first.isoformat(), last.isoformat()),
        )
        rows, failed = [], []
        if not granules:
            return rows, failed
        for handle in self._ea.open(granules):
            try:
                rows.append(read_point(handle, lat, lon))
            except Exception as err:
                failed.append(f"{type(err).__name__}: {err}"[:300])
            finally:
                try:
                    handle.close()
                except Exception:
                    pass
        return rows, failed


class GpmImerg:
    def __init__(self, regions, store, cache, fetcher=None, progress=True):
        self.regions = regions
        self.store = store
        self.cache = cache
        self.fetcher = fetcher or EarthaccessFetcher()
        self.progress = progress

    def _cache_key(self, lat, lon, month):
        name = getattr(self.fetcher, "short_name", SHORT_NAME)
        version = getattr(self.fetcher, "version", VERSION)
        return f"imerg:v{KEY_VERSION}:{name}:{version}:{lat:.4f}:{lon:.4f}:{month:%Y-%m}"

    def _month(self, lat, lon, first, last, final):
        key = self._cache_key(lat, lon, first)
        text = self.cache.get(key) if final else None
        if text is not None:
            return json.loads(text), []
        rows, failed = self.fetcher.month(first, last, lat, lon)
        rows = [r for r in rows if first.isoformat() <= r["date"] <= last.isoformat()]
        if final and not failed:
            self.cache.set(key, json.dumps(rows))
        return rows, failed

    def _save(self, key, first, rows):
        month = f"{first:%Y-%m}"
        frame = pd.DataFrame(rows, columns=["date", "cell_lat", "cell_lon", "precipitation", "precipitation_cnt", "unit"])
        daily = frame[["date", "precipitation", "precipitation_cnt"]].astype(
            {"date": "object", "precipitation": "float64", "precipitation_cnt": "float64"})
        daily.insert(0, "month", month)
        daily.insert(0, "point_key", key)
        cover = pd.DataFrame([{
            "point_key": key, "month": month, "days_found": int(len(frame)),
            "cell_lat": float(frame["cell_lat"].iloc[0]) if len(frame) else np.nan,
            "cell_lon": float(frame["cell_lon"].iloc[0]) if len(frame) else np.nan,
            "unit": frame["unit"].iloc[0] if len(frame) else None,
            "fetched_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        }])
        keys = {"point_key": key, "month": month}
        self.store.replace_many([(VALUES_TABLE, daily, keys), (COVERAGE_TABLE, cover, keys)])

    def coverage(self):
        return self.store.read_where(COVERAGE_TABLE)

    def series(self, lat, lon, start=DEFAULT_START, end=DEFAULT_END, refresh=False):
        start, end = _day(start), _day(end)
        if start > end or start < FIRST_DAY or end > date.today():
            raise ValueError(f"dates must satisfy {FIRST_DAY} <= start <= end <= today")
        lat, lon = round(float(lat), 4), round(float(lon), 4)
        where = self.regions.locate(lat, lon)
        key = f"{lat:.4f},{lon:.4f}"

        cover = self.store.read_where(COVERAGE_TABLE, point_key=key)
        stored = {} if cover.empty else {r.month: r for r in cover.itertuples()}
        months = list(_months(start, end))
        todo = [m for m in months if refresh or f"{m[0]:%Y-%m}" not in stored]

        fresh, failed, provisional, saved_now = [], [], False, set()
        cutoff = date.today() - timedelta(days=FINAL_AFTER_DAYS)
        bar = None
        if todo and self.progress:
            from tqdm.auto import tqdm

            bar = tqdm(total=len(todo), desc="IMERG months")
        for first, last in todo:
            final = last <= cutoff
            provisional = provisional or not final
            rows, bad = self._month(lat, lon, first, min(last, date.today()), final)
            failed += [f"{first:%Y-%m}: {b}" for b in bad]
            if final and not bad:
                self._save(key, first, rows)
                saved_now.add(f"{first:%Y-%m}")
            else:
                fresh += rows
            if bar is not None:
                bar.update(1)
        if bar is not None:
            bar.close()
        if failed:
            log.warning("%d IMERG files failed, see df.attrs['failed']", len(failed))

        saved = [f"{m[0]:%Y-%m}" for m in months if f"{m[0]:%Y-%m}" in stored or f"{m[0]:%Y-%m}" in saved_now]
        parts = []
        if saved:
            db = self.store.read_where(VALUES_TABLE, point_key=key, month=saved)
            parts.append(db[["date", "precipitation", "precipitation_cnt"]])
        if fresh:
            parts.append(pd.DataFrame(fresh)[["date", "precipitation", "precipitation_cnt"]])
        got = pd.concat(parts) if parts else pd.DataFrame(columns=["date", "precipitation", "precipitation_cnt"])
        got["date"] = pd.to_datetime(got["date"])
        got = got.drop_duplicates("date", keep="last").set_index("date").sort_index()

        days = pd.date_range(start, end, freq="D")
        out = got.reindex(days).rename_axis("date").reset_index()
        cell = self._cell(key, fresh)
        front = {"lat": lat, "lon": lon, **where, "cell_lat": cell[0], "cell_lon": cell[1]}
        for i, (col, val) in enumerate(front.items(), start=1):
            out.insert(i, col, val)
        out.attrs.update({
            "unit": cell[2],
            "missing_days": int(out["precipitation"].isna().sum()),
            "provisional": provisional,
            "failed": failed,
        })
        return out

    def _cell(self, key, fresh):
        if fresh:
            return fresh[0]["cell_lat"], fresh[0]["cell_lon"], fresh[0]["unit"]
        cover = self.store.read_where(COVERAGE_TABLE, point_key=key)
        cover = cover.dropna(subset=["cell_lat"]) if not cover.empty else cover
        if cover.empty:
            return None, None, None
        return float(cover["cell_lat"].iloc[0]), float(cover["cell_lon"].iloc[0]), cover["unit"].iloc[0]
