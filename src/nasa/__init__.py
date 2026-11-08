from .config import Settings
from .regions import OutsideNigeria, Regions
from .soil import SoilExplorer

__all__ = ["OutsideNigeria", "Regions", "Settings", "SoilExplorer", "main"]


def main() -> None:
    print("nasa soil tools, open the notebook in notebooks/ to explore")
