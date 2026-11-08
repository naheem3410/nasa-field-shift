import shutil
import subprocess
import time

import fakeredis
import pandas as pd
import pytest

from nasa.cache import NullCache, RedisCache, SqlStore, make_cache


def test_null_cache_is_passthrough():
    cache = NullCache()
    cache.set("a", "b")
    assert cache.get("a") is None


def test_make_cache_falls_back_when_redis_is_down():
    start = time.time()
    cache = make_cache("redis://localhost:6399/0", 60)
    assert isinstance(cache, NullCache)
    assert time.time() - start < 5


def test_make_cache_with_empty_url_is_null():
    assert isinstance(make_cache("", 60), NullCache)


def test_redis_cache_stores_with_ttl():
    client = fakeredis.FakeRedis(decode_responses=True)
    cache = RedisCache(client, ttl=100)
    cache.set("k", "v")
    assert cache.get("k") == "v"
    assert 0 < client.ttl("k") <= 100


def test_redis_cache_turns_into_passthrough_after_a_failure():
    class Broken:
        calls = 0

        def get(self, key):
            Broken.calls += 1
            raise ConnectionError("gone")

        def set(self, key, value, ex=None):
            raise ConnectionError("gone")

    cache = RedisCache(Broken(), ttl=10)
    assert cache.get("k") is None
    assert cache.available is False
    cache.set("k", "v")
    assert cache.get("k") is None
    assert Broken.calls == 1


@pytest.mark.skipif(shutil.which("redis-server") is None, reason="redis-server is not installed")
def test_real_redis_server_roundtrip(tmp_path):
    proc = subprocess.Popen(
        ["redis-server", "--port", "6398", "--save", "", "--appendonly", "no", "--dir", str(tmp_path)],
        stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
    )
    try:
        cache = None
        for _ in range(30):
            cache = make_cache("redis://localhost:6398/0", 60)
            if isinstance(cache, RedisCache):
                break
            time.sleep(0.2)
        assert isinstance(cache, RedisCache)
        cache.set("isda:test", "value")
        assert cache.get("isda:test") == "value"
    finally:
        proc.terminate()
        proc.wait(timeout=10)


def test_store_roundtrip_and_freshness(tmp_path):
    store = SqlStore(f"sqlite:///{(tmp_path / 'c.db').as_posix()}")
    df = pd.DataFrame({"a": [1.0, None, 3.0], "b": ["x", None, "z"], "c": [1, 2, 3]})
    assert not store.is_fresh("t", "sig1")
    store.write("t", df, "sig1", note="hello")
    assert store.is_fresh("t", "sig1")
    assert not store.is_fresh("t", "sig2")
    assert store.note("t") == "hello"
    back = store.read("t")
    pd.testing.assert_frame_equal(df, back, check_dtype=False)


def test_store_write_replaces_old_table(tmp_path):
    store = SqlStore(f"sqlite:///{(tmp_path / 'c.db').as_posix()}")
    store.write("t", pd.DataFrame({"a": [1, 2, 3]}), "one")
    store.write("t", pd.DataFrame({"a": [9]}), "two")
    assert store.read("t")["a"].tolist() == [9]
    assert store.is_fresh("t", "two")
    assert not store.is_fresh("t", "one")


def test_store_creates_missing_folder(tmp_path):
    target = tmp_path / "deep" / "er" / "c.db"
    SqlStore(f"sqlite:///{target.as_posix()}")
    assert target.exists()
