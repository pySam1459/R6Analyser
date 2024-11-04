from pathlib import Path
from pydantic import BaseModel, ConfigDict

from utils import load_json
from utils.constants import *


__all__ = [
    "Settings",
    "create_settings"
]

class Settings(BaseModel):
    defaults_filepath:  Path = DEFAULTS_PATH
    debug_filepath:     Path = DEBUG_PATH
    gsettings_filepath: Path = GAME_SETTINGS_PATH
    assets_path:        Path = ASSETS_PATH

    tessdata:           Path = DEFAULT_TESSDATA_PATH
    config_list_derive: bool = True

    model_config = ConfigDict(extra="ignore")


def create_settings(settings_path: Path) -> Settings:
    settings_json = load_json(settings_path)
    return Settings.model_validate(settings_json)
