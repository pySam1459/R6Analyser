from os import listdir
from pathlib import Path
from re import search, match
from pydantic import BaseModel, ConfigDict, Field, computed_field, model_validator, field_validator
from typing import Optional, Self, Any

from utils import Timestamp
from utils.keycheck import UserKeyData, validate_software_key, INVALID_KEY_REASONS
from utils.constants import *


__all__ = ["AnalyserArgs"]


def get_software_key_file() -> Path:
    files = [Path(f) for f in listdir(".")]
    for file in files:
        if file.is_file() and file.stem == "SOFTWARE_KEY":
            return file

    return SOFTWARE_KEY_FILE


def get_env_key() -> Optional[str]:
    file = get_software_key_file()
    with open(file, "r") as f_in:
        file_data = f_in.read()

    if (m := search(SOFTWARE_KEY_PATTERN, file_data)) is not None:
        return m.group(0)
    
    return None
    

class CliArgs(BaseModel):
    config_path:   Optional[Path] = None
    arg_key:       Optional[str]  = Field(default=None,
                                          pattern=SOFTWARE_KEY_PATTERN,
                                          validation_alias="key",
                                          exclude=True)
    verbose:       int                 = Field(default=1, ge=0, le=3)
    start:         Optional[Timestamp] = None

    sets_path:     Path = Field(default=SETTINGS_PATH, validation_alias="settings")

    check_regions: bool
    test_regions:  bool
    deps_check:    bool

    region_tool:   bool
    delay:         int  = 2
    display:       int  = 0

    model_config = ConfigDict(extra="ignore")

    @property
    def is_test(self) -> bool:
        return self.check_regions or self.test_regions
    
    @property
    def is_tool(self) -> bool:
        return self.region_tool


    @field_validator("config_path", mode="before")
    @classmethod
    def validate_config_path_before(cls, v: Any) -> Path:
        if isinstance(v, Path):
            return v
        elif not isinstance(v, str):
            raise ValueError(f"Invalid config_path type {type(v)}, must be Path/str")
        
        if not v.endswith(".json"):
            v += ".json"

        return Path(v)

    @field_validator("start", mode="before")
    @classmethod
    def parse_start(cls, v: Any) -> Optional[Timestamp]:
        if v is None or isinstance(v, Timestamp):
            return v
        elif not isinstance(v, str):
            return None

        if match(TIMESTAMP_PATTERN, v):
            return Timestamp.from_str(v)
        elif match(NUMBER_PATTERN, v) and int(v) >= 0:
            return Timestamp.from_int(int(v))
        return None

    @model_validator(mode="after")
    def validate_config_path_after(self) -> Self:
        if self.config_path is None:
            if self.deps_check:
                return self
            raise ValueError("No config provided!")

        if self.config_path.exists():
            if self.config_path.suffix == ".json":
                return self
            else:
                raise ValueError("Config files must end in .json")

        new_path = Path("configs") / self.config_path
        if new_path.exists():
            if new_path.suffix == ".json":
                self.config_path = new_path
                return self
            else:
                raise ValueError("Config files must end in .json")

        raise ValueError(f"Config file {self.config_path} cannot be found!")

    @model_validator(mode="after")
    def validate_settings(self) -> Self:
        if not self.sets_path.exists():
            raise ValueError(f"settings.json cannot be found {self.sets_path}")
        return self


class AnalyserArgs(CliArgs):
    software_key_data: Optional[UserKeyData] = None

    @computed_field
    @property
    def software_key(self) -> str:
        env_key = get_env_key()

        if self.arg_key is not None:
            return self.arg_key
        elif env_key is not None:
            return env_key
        else:
            raise ValueError("No Software Key Provided!")

    @model_validator(mode="after")
    def validate_software_key(self) -> Self:
        if self.region_tool or self.check_regions or self.test_regions or self.deps_check:
            return self

        key_data = validate_software_key(self.software_key)
        status_code = key_data.status_code
        if status_code == 200:
            self.software_key_data = UserKeyData.model_validate(key_data)
            return self
        else:
            reason = INVALID_KEY_REASONS[status_code]
            raise ValueError(f"Software Key provided is invalid!\nCode {status_code}: {reason}")
