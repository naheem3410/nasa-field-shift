import numpy as np

# metric crs for distances, equal area crs for areas
METRIC_CRS = 32632
AREA_CRS = 6933


def haversine_km(lat, lon, lats, lons):
    p = np.pi / 180
    a = (np.sin((lats - lat) * p / 2) ** 2
         + np.cos(lat * p) * np.cos(lats * p) * np.sin((lons - lon) * p / 2) ** 2)
    return 12742 * np.arcsin(np.sqrt(np.clip(a, 0, 1)))
