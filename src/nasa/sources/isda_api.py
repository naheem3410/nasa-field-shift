import json
import logging
import os
import time

import pandas as pd
import requests

from ..config import env_setting, url_path

log = logging.getLogger("nasa")

BASE_URL = "https://api.isda-africa.com"
LOGIN_PATH = "/login"
LAYERS_PATH = "/isdasoil/v2/layers"
PROPERTY_PATH = "/isdasoil/v2/soilproperty"
RETRY_CODES = {429, 502, 503, 504}
BASE_COLUMNS = [
    "lat", "lon", "state", "geopolitical_zone", "property", "description", "theme",
    "depth", "depth_unit", "value", "value_numeric", "unit", "value_type",
]


class IsdaApiError(RuntimeError):
    pass


class IsdaSoilApi:
    def __init__(self, regions, cache, session=None, base_url=None):
        self.regions = regions
        self.cache = cache
        self.session = session or requests.Session()
        self.base_url = (base_url or env_setting("ISDA_BASE_URL", BASE_URL)).rstrip("/")
        self.login_path = url_path(env_setting("ISDA_LOGIN_PATH", LOGIN_PATH))
        self.layers_path = url_path(env_setting("ISDA_LAYERS_PATH", LAYERS_PATH))
        self.property_path = url_path(env_setting("ISDA_PROPERTY_PATH", PROPERTY_PATH))
        self.token = None

    def _login(self):
        user = os.getenv("ISDA_USERNAME")
        password = os.getenv("ISDA_PASSWORD")
        if not user or not password:
            raise IsdaApiError("set ISDA_USERNAME and ISDA_PASSWORD in the .env file")
        form = {
            "grant_type": "password",
            "username": user,
            "password": password,
            "scope": "",
            "client_id": "string",
            "client_secret": "string",
        }
        resp = self.session.post(f"{self.base_url}{self.login_path}", data=form, timeout=30)
        if resp.status_code != 200:
            raise IsdaApiError(f"login failed with status {resp.status_code}")
        token = resp.json().get("access_token")
        if not token:
            raise IsdaApiError("login answered without an access_token")
        self.token = token

    def _get(self, path, params=None):
        relogged = False
        for attempt in range(4):
            if self.token is None:
                self._login()
            resp = self.session.get(
                f"{self.base_url}{path}", params=params,
                headers={"Authorization": f"Bearer {self.token}"}, timeout=30,
            )
            if resp.status_code == 401 and not relogged:
                self.token = None
                relogged = True
                continue
            if resp.status_code in RETRY_CODES and attempt < 3:
                time.sleep(2 ** attempt)
                continue
            break
        if resp.status_code != 200:
            raise IsdaApiError(f"{path} {params} gave status {resp.status_code}: {resp.text[:200]}")
        return resp.text

    def layers(self):
        key = "isda:v2:layers"
        text = self.cache.get(key)
        if text is None:
            text = self._get(self.layers_path)
            self.cache.set(key, text)
        rows = []
        for name, info in json.loads(text)["property"].items():
            rows.append({
                "property": name,
                "description": info.get("description"),
                "theme": info.get("theme"),
                "unit": info.get("unit"),
                "has_uncertainty": bool(info.get("uncertainty")),
                "value_type": (info.get("value") or {}).get("type"),
                "depths": list((info.get("depths") or {}).get("values") or []),
            })
        return pd.DataFrame(rows)

    def _value(self, prop, depth, lat, lon):
        key = f"isda:v2:{prop}:{depth}:{lat:.5f}:{lon:.5f}"
        text = self.cache.get(key)
        if text is None:
            params = {"lon": f"{lon:.5f}", "lat": f"{lat:.5f}", "depth": depth, "property": prop}
            text = self._get(self.property_path, params)
            self.cache.set(key, text)
        return text

    def _parse(self, text, prop, lay, where, lat, lon):
        entries = json.loads(text).get("property", {}).get(prop)
        if not entries:
            raise IsdaApiError(f"no values for {prop} in the response")
        rows = []
        for entry in entries:
            value = entry.get("value") or {}
            depth = entry.get("depth") or {}
            row = {
                "lat": lat, "lon": lon, **where,
                "property": prop,
                "description": lay["description"],
                "theme": lay["theme"],
                "depth": depth.get("value"),
                "depth_unit": depth.get("unit"),
                "value": value.get("value"),
                "unit": value.get("unit"),
                "value_type": value.get("type"),
            }
            for band in entry.get("uncertainty") or []:
                level = str(band.get("confidence_interval", "")).rstrip("%")
                row[f"lower_{level}"] = band.get("lower_bound")
                row[f"upper_{level}"] = band.get("upper_bound")
            rows.append(row)
        return rows

    def point(self, lat, lon, properties=None, depths=None):
        lat, lon = round(float(lat), 5), round(float(lon), 5)
        where = self.regions.locate(lat, lon)
        layers = self.layers()
        if properties:
            unknown = sorted(set(properties) - set(layers["property"]))
            if unknown:
                raise ValueError(f"unknown properties: {unknown}")
        rows, failed = [], []
        for _, lay in layers.iterrows():
            if properties and lay["property"] not in properties:
                continue
            for depth in lay["depths"]:
                if depths and depth not in depths:
                    continue
                try:
                    text = self._value(lay["property"], depth, lat, lon)
                    rows.extend(self._parse(text, lay["property"], lay, where, lat, lon))
                except IsdaApiError as err:
                    failed.append({"property": lay["property"], "depth": depth, "error": str(err)})
        if rows:
            df = pd.DataFrame(rows)
            df["value_numeric"] = pd.to_numeric(df["value"], errors="coerce")
            extra = sorted(c for c in df.columns if c not in BASE_COLUMNS)
            df = df.reindex(columns=BASE_COLUMNS + extra)
        else:
            df = pd.DataFrame(columns=BASE_COLUMNS)
        df.attrs["failed"] = failed
        if failed:
            log.warning("%d iSDAsoil requests failed, see df.attrs['failed']", len(failed))
        return df
