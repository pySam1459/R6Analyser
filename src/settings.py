from pathlib import Path
from pydantic import BaseModel, ConfigDict, Field, field_validator
from typing import Any

from utils import load_json
from utils.constants import *


__all__ = [
    "Settings",
    "create_settings"
]

class Settings(BaseModel):
    defaults_filepath:   Path = DEFAULTS_PATH
    debug_filepath:      Path = DEBUG_PATH

    tessdata:            Path = DEFAULT_TESSDATA_PATH
    languages:      list[str] = Field(default_factory=lambda: [DEFAULT_LANGUAGE])

    config_list_derive:  bool = True

    model_config = ConfigDict(extra="ignore")

    # @field_validator("languages")
    # @classmethod
    # def validate_languages(cls, v: Any) -> list[str]:
    #     if isinstance(v, str) and v in LANGUAGES:
    #         return [v]
    #     elif isinstance(v, list) and all([lang in get_languages() for lang in v]):
    #         return v

    #     raise ValueError(f"Invalid language list")

    @field_validator("config_list_derive")
    @classmethod
    def validate_cld(cls, v: Any) -> bool:
        if isinstance(v, bool):
            return v
        elif isinstance(v, str):
            if v.lower() in TRUTHY_STRINGS:
                return True
            elif v.lower() in FALSEY_STRINGS:
                return False
        raise ValueError(f"Invalid config_list_derive value: {v}")


def create_settings(settings_path: Path) -> Settings:
    settings_json = load_json(settings_path)
    return Settings.model_validate(settings_json)
