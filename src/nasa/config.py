import os
from dataclasses import dataclass
from pathlib import Path

from dotenv import load_dotenv


def find_root(start=None):
    here = Path(start or Path.cwd()).resolve()
    for folder in [here, *here.parents]:
        if (folder / "pyproject.toml").exists():
            return folder
    return here


def _path(root, name, default):
    value = os.getenv(name)
    path = Path(value) if value else Path(default)
    return path if path.is_absolute() else root / path


@dataclass
class Settings:
    root: Path
    afsp_dir: Path
    isda_csv: Path
    soil_map_dir: Path
    boundaries_path: Path
    database_url: str
    redis_url: str
    redis_ttl: int

    @classmethod
    def load(cls, root=None):
        root = Path(root).resolve() if root else find_root()
        load_dotenv(root / ".env")
        db_file = _path(root, "SQLITE_PATH", "data/cache/soil.db")
        return cls(
            root=root,
            afsp_dir=_path(root, "AFSP_DBF_DIR", "data/AF-AfSP1.2/AfSP012Qry_ISRIC/GIS_Dbf"),
            isda_csv=_path(root, "ISDA_CSV_PATH", "data/iSDA_soil_data.csv"),
            soil_map_dir=_path(root, "SOIL_MAP_DIR", "data/mendeley-data/digital-soil-map/soil types"),
            boundaries_path=_path(root, "BOUNDARIES_PATH", "data/boundaries/geoBoundaries-NGA-ADM1.geojson"),
            database_url=os.getenv("DATABASE_URL") or "sqlite:///" + db_file.as_posix(),
            redis_url=os.getenv("REDIS_URL", "redis://localhost:6379/0"),
            redis_ttl=int(os.getenv("API_CACHE_DAYS", "30")) * 86400,
        )
