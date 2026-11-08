import json
import logging
import re
import time
from datetime import date, datetime, timedelta, timezone

import pandas as pd
import requests

log = logging.getLogger("nasa")

BASE_URL = "https://power.larc.nasa.gov/api/temporal/daily/point"
RETRY_CODES = {429, 502, 503, 504}
VALUES_TABLE = "power_daily"
COVERAGE_TABLE = "power_coverage"
KEY_VERSION = "1"
DEFAULT_START = "2001-01-01"
DEFAULT_END = "2025-12-31"
DEFAULT_PARAMETERS = [
    "T2M", "T2M_MAX", "T2M_MIN", "T2MDEW", "RH2M", "WS2M",
    "PRECTOTCORR", "ALLSKY_SFC_SW_DWN", "GWETTOP", "GWETROOT",
]
FIRST_DAY = date(1981, 1, 1)
MAX_PARAMETERS = 20
CHUNK_YEARS = 5
# POWER replaces its recent values a few months later, so newer days are not stored
FINAL_AFTER_DAYS = 100


class NasaPowerError(RuntimeError):
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


def _chunks(start, end):
    while start <= end:
        stop = min(date(start.year + CHUNK_YEARS - 1, 12, 31), end)
        yield start, stop
        start = stop + timedelta(days=1)


class NasaPower:
    def __init__(self, regions, store, cache, session=None, base_url=BASE_URL):
        self.regions = regions
        self.store = store
        self.cache = cache
        self.session = session or requests.Session()
        self.base_url = base_url

    def _params(self, parameters):
        names = DEFAULT_PARAMETERS if parameters is None else list(parameters)
        names = list(dict.fromkeys(str(n).strip().upper() for n in names))
        if not names or len(names) > MAX_PARAMETERS:
            raise ValueError(f"ask for 1 to {MAX_PARAMETERS} parameters")
        bad = [n for n in names if not re.fullmatch(r"[A-Z0-9_]+", n)]
        if bad:
            raise ValueError(f"not a POWER parameter name: {bad}")
        return names

    def _get(self, params):
        for attempt in range(4):
            resp = self.session.get(self.base_url, params=params, timeout=60)
            if resp.status_code in RETRY_CODES and attempt < 3:
                time.sleep(2 ** attempt)
                continue
            break
        if resp.status_code != 200:
            raise NasaPowerError(f"POWER gave status {resp.status_code}: {resp.text[:300]}")
        return resp.text

    def _request(self, lat, lon, first, last, names):
        key = f"power:v{KEY_VERSION}:{lat:.4f}:{lon:.4f}:{first}:{last}:{','.join(sorted(names))}"
        final = last <= date.today() - timedelta(days=FINAL_AFTER_DAYS)
        text = self.cache.get(key) if final else None
        if text is None:
            text = self._get({
                "parameters": ",".join(names), "community": "AG",
                "longitude": f"{lon:.4f}", "latitude": f"{lat:.4f}",
                "start": first.strftime("%Y%m%d"), "end": last.strftime("%Y%m%d"), "format": "JSON",
            })
            if final:
                self.cache.set(key, text)
        try:
            return json.loads(text)
        except ValueError as err:
            raise NasaPowerError(f"POWER answered with something that is not JSON: {text[:200]}") from err

    def _read(self, js, names, parts, meta):
        body = (js.get("properties") or {}).get("parameter") or {}
        fill = float((js.get("header") or {}).get("fill_value", -999))
        info = js.get("parameters") or {}
        for name in names:
            if name not in body:
                raise NasaPowerError(f"POWER did not return {name}, messages: {js.get('messages')}")
            ser = pd.Series(body[name], dtype=float)
            parts[name].append(ser.mask(ser == fill))
            meta["units"][name] = (info.get(name) or {}).get("units")
        coords = (js.get("geometry") or {}).get("coordinates") or []
        if len(coords) >= 3:
            meta["elevation_m"] = float(coords[2])

    def _download(self, lat, lon, start, end, names):
        parts = {n: [] for n in names}
        meta = {"units": {}}
        for first, last in _chunks(start, end):
            self._read(self._request(lat, lon, first, last, names), names, parts, meta)
        fresh = {}
        for name, pieces in parts.items():
            ser = pd.concat(pieces)
            ser.index = pd.to_datetime(ser.index, format="%Y%m%d")
            fresh[name] = ser[~ser.index.duplicated()].sort_index()
        return fresh, meta

    def _save(self, key, fresh, meta, start, end):
        stamp = datetime.now(timezone.utc).isoformat(timespec="seconds")
        items = []
        for name, ser in fresh.items():
            values = pd.DataFrame({
                "point_key": key, "parameter": name,
                "date": ser.index.strftime("%Y-%m-%d"), "value": ser.to_numpy(),
            })
            cover = pd.DataFrame([{
                "point_key": key, "parameter": name, "unit": meta["units"].get(name),
                "start": start.isoformat(), "end": end.isoformat(),
                "elevation_m": meta.get("elevation_m"), "fetched_at": stamp,
            }])
            items.append((VALUES_TABLE, values, {"point_key": key, "parameter": name}))
            items.append((COVERAGE_TABLE, cover, {"point_key": key, "parameter": name}))
        self.store.replace_many(items)

    def coverage(self):
        return self.store.read_where(COVERAGE_TABLE)

    def series(self, lat, lon, start=DEFAULT_START, end=DEFAULT_END, parameters=None, refresh=False):
        start, end = _day(start), _day(end)
        if start > end or start < FIRST_DAY or end > date.today():
            raise ValueError(f"dates must satisfy {FIRST_DAY} <= start <= end <= today")
        names = self._params(parameters)
        lat, lon = round(float(lat), 4), round(float(lon), 4)
        where = self.regions.locate(lat, lon)
        key = f"{lat:.4f},{lon:.4f}"

        cover = self.store.read_where(COVERAGE_TABLE, point_key=key)
        have = {}
        if not cover.empty:
            have = {r.parameter: r for r in cover.itertuples() if r.start <= start.isoformat() and r.end >= end.isoformat()}
        todo = names if refresh else [n for n in names if n not in have]
        cached = [n for n in names if n not in todo]

        parts, meta = {}, {"units": {}}
        provisional = end > date.today() - timedelta(days=FINAL_AFTER_DAYS)
        if todo:
            fresh, meta = self._download(lat, lon, start, end, todo)
            parts.update(fresh)
            if not provisional:
                self._save(key, fresh, meta, start, end)
        if cached:
            rows = self.store.read_where(VALUES_TABLE, point_key=key, parameter=cached)
            rows = rows[(rows["date"] >= start.isoformat()) & (rows["date"] <= end.isoformat())]
            for name, grp in rows.groupby("parameter"):
                parts[name] = pd.Series(grp["value"].to_numpy(dtype=float), index=pd.to_datetime(grp["date"]))
            first = have[cached[0]]
            for name in cached:
                meta["units"][name] = have[name].unit
            meta.setdefault("elevation_m", first.elevation_m)

        days = pd.date_range(start, end, freq="D")
        wide = pd.DataFrame({n: parts[n].reindex(days) for n in names}, index=days)
        out = wide.rename_axis("date").reset_index()
        front = {"lat": lat, "lon": lon, **where}
        for i, (col, val) in enumerate(front.items(), start=1):
            out.insert(i, col, val)
        out.attrs.update({
            "units": {n: meta["units"].get(n) for n in names},
            "elevation_m": meta.get("elevation_m"),
            "missing_days": {n: int(wide[n].isna().sum()) for n in names},
            "provisional": provisional,
        })
        return out
