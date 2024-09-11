from pathlib import Path
from easyocr.config import all_lang_list
from pydantic import BaseModel, ConfigDict, Field, field_validator
from typing import Any

from utils import load_json
from utils.enums import OCREngineType
from utils.constants import *


__all__ = [
    "Settings",
    "create_settings"
]


def __validate_language_list(lang_list: list[str]) -> list[str]:
    for lang_code in lang_list:
        if lang_code not in all_lang_list:
            raise ValueError(f"Invalid language code {lang_code}")
    return lang_list


class Settings(BaseModel):
    defaults_filepath:  Path  = DEFAULTS_PATH
    debug_filepath:     Path  = DEBUG_PATH

    ocr_engine: OCREngineType = OCREngineType.EASYOCR
    languages:     list[str]  = Field(default_factory=lambda: [DEFAULT_LANGUAGE])

    config_list_derive: bool  = True

    model_config = ConfigDict(extra="ignore")

    @field_validator("languages")
    @classmethod
    def validate_languages(cls, v: Any) -> list[str]:
        if isinstance(v, str):
            return __validate_language_list([v])
        elif isinstance(v, list):
            return __validate_language_list(v)

        raise ValueError(f"Invalid language list")

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
