from pathlib import Path
from pydantic import BaseModel
from typing import Any, TypeVar, Callable

from settings import Settings
from utils import load_json, recursive_union


def load_config_json(config_path: Path) -> dict[str,Any] | list[dict[str,Any]]:
    cfg_json = load_json(config_path)
    if isinstance(cfg_json, dict):
        return cfg_json | {"config_path": config_path}
    elif isinstance(cfg_json, list):
        return [el | {"config_path": config_path} for el in cfg_json]

    raise ValueError(f"Invalid config file {config_path}")


T = TypeVar('T', bound=BaseModel)
def validate_config(validate_func: Callable[[Any], T], config_path: Path, settings: Settings) -> T | list[T]:
    cfg_json = load_config_json(config_path)

    if isinstance(cfg_json, dict):
        return validate_func(cfg_json)

    if not settings.config_list_derive: ## all configs in list are full
        return [validate_func(el) for el in cfg_json]

    first_config = validate_func(cfg_json[0])
    first_data = first_config.model_dump()
    return [first_config] + [validate_func(recursive_union(first_data, el)) for el in cfg_json[1:]]
