import json
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
FIXTURES = Path(__file__).parent / "fixtures"

DATA_READY = all(
    p.exists() for p in [
        ROOT / "data" / "iSDA_soil_data.csv",
        ROOT / "data" / "AF-AfSP1.2" / "AfSP012Qry_ISRIC" / "GIS_Dbf" / "AfSP012Qry_Layers.dbf",
        ROOT / "data" / "mendeley-data" / "digital-soil-map" / "soil types" / "soil types.shp",
    ]
)
needs_data = pytest.mark.skipif(not DATA_READY, reason="local data files are not in data/")

CARBON = {
    "property": {"carbon_organic": [{
        "value": {"unit": "g/kg", "type": "float", "value": 19.1},
        "depth": {"value": "0-20", "unit": "cm"},
        "uncertainty": [
            {"confidence_interval": "50%", "lower_bound": 17.8, "upper_bound": 20.5},
            {"confidence_interval": "68%", "lower_bound": 17.2, "upper_bound": 21.2},
            {"confidence_interval": "90%", "lower_bound": 16.0, "upper_bound": 22.7},
        ],
    }]}
}
CROP = {
    "property": {"crop_cover_2019": [{
        "value": {"unit": "%", "type": "integer", "value": 36},
        "depth": {"value": "0", "unit": "cm"},
        "uncertainty": None,
    }]}
}


class FakeResponse:
    def __init__(self, status, body):
        self.status_code = status
        self.text = body if isinstance(body, str) else json.dumps(body)

    def json(self):
        return json.loads(self.text)


class FakeSession:
    def __init__(self):
        self.layers = json.loads((FIXTURES / "isda_layers.json").read_text())
        self.posts = []
        self.gets = []
        self.status_for = {}
        self.once = {}

    def post(self, url, data=None, timeout=None):
        self.posts.append((url, data))
        return FakeResponse(200, {"access_token": f"token{len(self.posts)}", "token_type": "bearer"})

    def get(self, url, params=None, headers=None, timeout=None):
        self.gets.append((url, params, headers))
        if url.endswith("/layers"):
            return FakeResponse(200, self.layers)
        key = (params["property"], params["depth"])
        if key in self.once:
            return FakeResponse(self.once.pop(key), "temporary")
        if key in self.status_for:
            return FakeResponse(self.status_for[key], "nope")
        return FakeResponse(200, self.make(params["property"], params["depth"]))

    def make(self, prop, depth):
        if prop == "carbon_organic" and depth == "0-20":
            return CARBON
        if prop == "crop_cover_2019":
            return CROP
        info = self.layers["property"][prop]
        kind = info["value"]["type"]
        value = {"float": 2.5, "integer": 7}.get(kind, "loam")
        bands = None
        if info["uncertainty"]:
            bands = [{"confidence_interval": "90%", "lower_bound": 1.0, "upper_bound": 4.0}]
        return {"property": {prop: [{
            "value": {"unit": info["unit"], "type": kind, "value": value},
            "depth": {"value": depth, "unit": "cm"},
            "uncertainty": bands,
        }]}}

    def soil_gets(self):
        return [g for g in self.gets if g[0].endswith("/soilproperty")]


@pytest.fixture
def session():
    return FakeSession()


@pytest.fixture(autouse=True)
def fake_credentials(monkeypatch):
    monkeypatch.setenv("ISDA_USERNAME", "tester@example.com")
    monkeypatch.setenv("ISDA_PASSWORD", "not-a-real-password")


@pytest.fixture(scope="session")
def regions():
    from nasa.regions import Regions

    return Regions(ROOT / "data" / "boundaries" / "geoBoundaries-NGA-ADM1.geojson")


@pytest.fixture(scope="session")
def settings():
    from nasa.config import Settings

    return Settings.load(ROOT)
