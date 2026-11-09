import pytest

from nasa.cache import NullCache
from nasa.sources import IsdaSoilApi, NasaPower
from nasa.sources.nasa_power import NasaPowerError

from .conftest import FakeResponse

NAMES = ["ISDA_BASE_URL", "ISDA_LOGIN_PATH", "ISDA_LAYERS_PATH", "ISDA_PROPERTY_PATH", "POWER_BASE_URL", "POWER_DAILY_PATH"]


@pytest.fixture(autouse=True)
def clean_env(monkeypatch):
    for name in NAMES:
        monkeypatch.delenv(name, raising=False)


class UrlRecorder:
    def __init__(self):
        self.urls = []

    def get(self, url, params=None, timeout=None):
        self.urls.append(url)
        return FakeResponse(500, "stop here")


def isda(regions, session):
    return IsdaSoilApi(regions, NullCache(), session=session)


def test_isda_defaults_are_the_old_addresses(regions, session):
    api = isda(regions, session)
    api.layers()
    api.point(9.0765, 7.3986, properties=["ph"], depths=["0-20"])
    assert session.posts[0][0] == "https://api.isda-africa.com/login"
    urls = {g[0] for g in session.gets}
    assert urls == {"https://api.isda-africa.com/isdasoil/v2/layers", "https://api.isda-africa.com/isdasoil/v2/soilproperty"}


def test_isda_env_overrides_every_part(regions, session, monkeypatch):
    monkeypatch.setenv("ISDA_BASE_URL", "https://example.test/")
    monkeypatch.setenv("ISDA_LOGIN_PATH", "auth/in")
    monkeypatch.setenv("ISDA_LAYERS_PATH", "/v9/layers")
    monkeypatch.setenv("ISDA_PROPERTY_PATH", "/v9/prop")
    api = isda(regions, session)
    api.layers()
    api.point(9.0765, 7.3986, properties=["ph"], depths=["0-20"])
    assert session.posts[0][0] == "https://example.test/auth/in"
    assert {g[0] for g in session.gets} == {"https://example.test/v9/layers", "https://example.test/v9/prop"}


def test_isda_blank_env_values_fall_back(regions, session, monkeypatch):
    monkeypatch.setenv("ISDA_BASE_URL", "  ")
    monkeypatch.setenv("ISDA_LOGIN_PATH", "")
    api = isda(regions, session)
    assert api.base_url == "https://api.isda-africa.com" and api.login_path == "/login"


def test_isda_explicit_argument_beats_env(regions, session, monkeypatch):
    monkeypatch.setenv("ISDA_BASE_URL", "https://example.test")
    api = IsdaSoilApi(regions, NullCache(), session=session, base_url="https://other.test")
    assert api.base_url == "https://other.test"


def power(regions, store=None, session=None, **kw):
    return NasaPower(regions, store, NullCache(), session=session, **kw)


def test_power_default_is_the_old_address(regions):
    assert power(regions).base_url == "https://power.larc.nasa.gov/api/temporal/daily/point"


def test_power_env_overrides_and_is_used_for_requests(regions, monkeypatch):
    monkeypatch.setenv("POWER_BASE_URL", "https://example.test/")
    monkeypatch.setenv("POWER_DAILY_PATH", "v2/daily")
    rec = UrlRecorder()
    src = power(regions, session=rec)
    assert src.base_url == "https://example.test/v2/daily"
    with pytest.raises(NasaPowerError):
        src._get({})
    assert rec.urls == ["https://example.test/v2/daily"]


def test_power_blank_env_falls_back(regions, monkeypatch):
    monkeypatch.setenv("POWER_BASE_URL", "")
    monkeypatch.setenv("POWER_DAILY_PATH", " ")
    assert power(regions).base_url == "https://power.larc.nasa.gov/api/temporal/daily/point"


def test_power_explicit_argument_is_the_whole_address(regions, monkeypatch):
    monkeypatch.setenv("POWER_BASE_URL", "https://example.test")
    assert power(regions, base_url="https://full.test/x/y").base_url == "https://full.test/x/y"
