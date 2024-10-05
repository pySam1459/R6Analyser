from pathlib import Path
from pydantic import BaseModel, ConfigDict, Field, computed_field, model_validator, field_validator
from pydantic_settings import BaseSettings, SettingsConfigDict
from typing import Optional, Self, Any

from .keycheck import UserKeyData, validate_software_key
from .constants import SOFTWARE_KEY_PATTERN, SOFTWARE_KEY_FILE, SETTINGS_PATH


__all__ = ["AnalyserArgs"]


class Env(BaseSettings):
    env_key:    Optional[str] = Field(default=None,
                                      pattern=SOFTWARE_KEY_PATTERN,
                                      validation_alias="software_key",
                                      exclude=True)

    model_config = SettingsConfigDict(env_file=SOFTWARE_KEY_FILE, env_file_encoding="utf-8")


class CliArgs(BaseModel):
    config_path:   Optional[Path] = None
    arg_key:       Optional[str]  = Field(default=None,
                                          pattern=SOFTWARE_KEY_PATTERN,
                                          validation_alias="key",
                                          exclude=True)
    verbose:  int       = Field(default=1, ge=0, le=3)
    sets_path: Path     = Field(default=SETTINGS_PATH, validation_alias="settings")

    check_regions: bool
    test_regions:  bool
    deps_check:    bool

    region_tool:   bool
    delay:         int  = 2
    display:       int  = 0

    model_config = ConfigDict(extra="ignore")

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


class AnalyserArgs(Env, CliArgs):
    software_key_data: Optional[UserKeyData] = None

    @computed_field
    @property
    def software_key(self) -> str:
        if self.arg_key is not None:
            return self.arg_key
        elif self.env_key is not None:
            return self.env_key
        else:
            raise ValueError("No Software Key Provided!")

    @model_validator(mode='after')
    def validate_software_key(self) -> Self:
        if self.region_tool or self.check_regions or self.test_regions or self.deps_check:
            return self

        key_data = validate_software_key(self.software_key)
        status_code = key_data.status_code
        if status_code == 200:
            self.software_key_data = UserKeyData.model_validate(key_data)
            return self
        else:
            raise ValueError(f"Software Key provided is invalid! Status code {status_code}")
