import logging

log = logging.getLogger("nasa")


class NullCache:
    available = False

    def get(self, key):
        return None

    def set(self, key, value, ttl=None):
        return None


class RedisCache:
    def __init__(self, client, ttl):
        self.client = client
        self.ttl = ttl
        self.available = True

    def _fail(self, err):
        if self.available:
            log.warning("redis stopped answering (%s), api calls will not be cached", err)
        self.available = False

    def get(self, key):
        if not self.available:
            return None
        try:
            return self.client.get(key)
        except Exception as err:
            self._fail(err)
            return None

    def set(self, key, value, ttl=None):
        if not self.available:
            return None
        try:
            self.client.set(key, value, ex=ttl or self.ttl)
        except Exception as err:
            self._fail(err)


def make_cache(url, ttl):
    if not url:
        return NullCache()
    try:
        import redis

        client = redis.Redis.from_url(url, socket_connect_timeout=1, socket_timeout=2, decode_responses=True)
        client.ping()
    except Exception as err:
        log.warning("redis is not reachable (%s), api calls will not be cached", err)
        return NullCache()
    return RedisCache(client, ttl)
