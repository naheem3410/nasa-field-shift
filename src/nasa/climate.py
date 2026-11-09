from .cache import SqlStore, make_cache
from .config import Settings
from .regions import Regions
from .sources import GpmImerg, NasaPower


class Climate:
    def __init__(self, settings=None, session=None, cache=None, imerg_fetcher=None):
        self.settings = settings or Settings.load()
        self.store = SqlStore(self.settings.database_url)
        self.regions = Regions(self.settings.boundaries_path)
        if cache is None:
            cache = make_cache(self.settings.redis_url, self.settings.redis_ttl)
        self.cache = cache
        self.power = NasaPower(self.regions, self.store, self.cache, session=session)
        self.imerg = GpmImerg(self.regions, self.store, self.cache, fetcher=imerg_fetcher)
