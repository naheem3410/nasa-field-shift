import logging

import fakeredis
import pytest

from nasa.cache import NullCache, RedisCache
from nasa.regions import OutsideNigeria
from nasa.sources import IsdaApiError, IsdaSoilApi

ABUJA = (9.0765, 7.3986)


@pytest.fixture(autouse=True)
def no_sleep(monkeypatch):
    monkeypatch.setattr("nasa.sources.isda_api.time.sleep", lambda s: None)


def make(regions, session, cache=None):
    return IsdaSoilApi(regions, cache or NullCache(), session=session)


def pairs(session):
    return sum(len(p["depths"]["values"]) for p in session.layers["property"].values())


def test_layers_table(regions, session):
    layers = make(regions, session).layers()
    assert len(layers) == len(session.layers["property"])
    assert {"property", "description", "theme", "unit", "has_uncertainty", "value_type", "depths"} <= set(layers.columns)
    ph = layers[layers["property"] == "ph"].iloc[0]
    assert ph["depths"] == ["0-20", "20-50"]
    assert bool(ph["has_uncertainty"]) is True


def test_point_returns_every_property_and_depth(regions, session):
    df = make(regions, session).point(*ABUJA)
    assert len(df) == pairs(session)
    assert df.attrs["failed"] == []
    assert set(df["state"]) == {"Federal Capital Territory"}
    assert set(df["geopolitical_zone"]) == {"North Central"}
    assert len(session.soil_gets()) == pairs(session)


def test_values_and_uncertainty_are_parsed(regions, session):
    df = make(regions, session).point(*ABUJA, properties=["carbon_organic"], depths=["0-20"])
    row = df.iloc[0]
    assert row["value"] == 19.1 and row["value_numeric"] == 19.1
    assert row["unit"] == "g/kg" and row["depth"] == "0-20" and row["depth_unit"] == "cm"
    assert (row["lower_50"], row["upper_50"]) == (17.8, 20.5)
    assert (row["lower_68"], row["upper_68"]) == (17.2, 21.2)
    assert (row["lower_90"], row["upper_90"]) == (16.0, 22.7)


def test_layer_without_uncertainty_has_no_bounds(regions, session):
    df = make(regions, session).point(*ABUJA, properties=["crop_cover_2019"])
    row = df.iloc[0]
    assert row["value"] == 36 and row["unit"] == "%" and row["value_type"] == "integer"
    assert df.filter(like="lower_").empty


def test_text_values_stay_text_and_numeric_is_nan(regions, session):
    df = make(regions, session).point(*ABUJA, properties=["texture_class"])
    assert df["value"].tolist() == ["loam", "loam"]
    assert df["value_numeric"].isna().all()


def test_request_parameters(regions, session):
    make(regions, session).point(*ABUJA, properties=["ph"], depths=["0-20"])
    url, params, headers = session.soil_gets()[0]
    assert url == "https://api.isda-africa.com/isdasoil/v2/soilproperty"
    assert params == {"lon": "7.39860", "lat": "9.07650", "depth": "0-20", "property": "ph"}
    assert headers["Authorization"] == "Bearer token1"


def test_login_form_matches_the_documented_call(regions, session, caplog):
    caplog.set_level(logging.DEBUG)
    make(regions, session).point(*ABUJA, properties=["ph"])
    url, form = session.posts[0]
    assert url == "https://api.isda-africa.com/login"
    assert form == {
        "grant_type": "password", "username": "tester@example.com", "password": "not-a-real-password",
        "scope": "", "client_id": "string", "client_secret": "string",
    }
    assert "not-a-real-password" not in caplog.text
    assert len(session.posts) == 1


def test_missing_credentials_message(regions, session, monkeypatch):
    monkeypatch.delenv("ISDA_PASSWORD")
    with pytest.raises(IsdaApiError, match="ISDA_USERNAME and ISDA_PASSWORD"):
        make(regions, session).point(*ABUJA, properties=["ph"])


def test_outside_nigeria_makes_no_http_call(regions, session):
    with pytest.raises(OutsideNigeria):
        make(regions, session).point(-0.7196, 35.24)
    assert session.gets == [] and session.posts == []


def test_unknown_property_is_rejected(regions, session):
    with pytest.raises(ValueError, match="unknown properties"):
        make(regions, session).point(*ABUJA, properties=["ph", "not_real"])


def test_expired_token_logs_in_again(regions, session):
    session.once[("ph", "0-20")] = 401
    df = make(regions, session).point(*ABUJA, properties=["ph"], depths=["0-20"])
    assert len(df) == 1 and len(session.posts) == 2
    assert session.soil_gets()[-1][2]["Authorization"] == "Bearer token2"


def test_temporary_server_error_is_retried(regions, session):
    session.once[("ph", "0-20")] = 503
    df = make(regions, session).point(*ABUJA, properties=["ph"], depths=["0-20"])
    assert len(df) == 1 and df.attrs["failed"] == []
    assert len(session.soil_gets()) == 2


def test_permanent_failure_is_reported_not_hidden(regions, session, caplog):
    session.status_for[("ph", "20-50")] = 404
    df = make(regions, session).point(*ABUJA, properties=["ph"])
    assert len(df) == 1
    failed = df.attrs["failed"]
    assert len(failed) == 1 and failed[0]["property"] == "ph" and failed[0]["depth"] == "20-50"
    assert "404" in failed[0]["error"]


def test_empty_result_still_has_columns(regions, session):
    session.status_for[("ph", "0-20")] = 404
    session.status_for[("ph", "20-50")] = 404
    df = make(regions, session).point(*ABUJA, properties=["ph"])
    assert df.empty and "value_numeric" in df.columns and len(df.attrs["failed"]) == 2


def test_redis_cache_serves_second_call_without_http(regions, session):
    cache = RedisCache(fakeredis.FakeRedis(decode_responses=True), ttl=1000)
    api = make(regions, session, cache)
    first = api.point(*ABUJA)
    calls = len(session.gets)
    second = api.point(*ABUJA)
    assert len(session.gets) == calls
    assert first.equals(second)


def test_without_cache_every_call_hits_the_api(regions, session):
    api = make(regions, session)
    api.point(*ABUJA, properties=["ph"])
    api.point(*ABUJA, properties=["ph"])
    assert len(session.soil_gets()) == 4


def test_coordinates_are_rounded_to_five_decimals(regions, session):
    df = make(regions, session).point(9.076500004, 7.398600004, properties=["ph"], depths=["0-20"])
    assert df.iloc[0]["lat"] == 9.0765 and df.iloc[0]["lon"] == 7.3986
