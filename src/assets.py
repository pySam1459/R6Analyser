import cv2
import numpy as np
from pathlib import Path
from pydantic import BaseModel

from utils import load_file


__all__ = [
    "Assets"
]


class _AssetMap(BaseModel):
    atkside_template: str
    headshot_mask: str
    timer_inf: str


class Assets:
    def __init__(self, assets_path: Path) -> None:
        asset_map_file = assets_path / "asset_map.json"
        json_data = load_file(asset_map_file)
        self.asset_map = _AssetMap.model_validate_json(json_data)

        self.__originals = {name: Assets._load_asset(assets_path / file)
                            for name, file in self.asset_map.model_dump().items()}
        self.__cache = {}

    @staticmethod
    def _load_asset(path: Path) -> np.ndarray:
        if not path.exists():
            raise ValueError(f"Asset path {path} does not exist!")
        
        return cv2.imread(str(path), cv2.IMREAD_GRAYSCALE)
    
    def __getitem__(self, key: str) -> np.ndarray:
        return self.__cache.get(key, self.__originals[key])
