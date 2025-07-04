from pathlib import Path
from pydantic import BaseModel, BeforeValidator
from typing import Any, Annotated, TypeVar, TypeAlias, Callable

import settings
from utils import load_json, recursive_union, BBox_t
from utils.enums import CaptureMode


def load_config_json(config_path: Path) -> dict[str,Any] | list[dict[str,Any]]:
    cfg_json = load_json(config_path)
    if isinstance(cfg_json, dict):
        return cfg_json | {"config_path": config_path}
    elif isinstance(cfg_json, list):
        return [el | {"config_path": config_path} for el in cfg_json]

    raise ValueError(f"Invalid config file {config_path}")


T = TypeVar('T', bound=BaseModel)
def validate_config(validate_func: Callable[[Any], T], config_path: Path) -> T | list[T]:
    cfg_json = load_config_json(config_path)

    if isinstance(cfg_json, dict):
        return validate_func(cfg_json)

    if not settings.SETTINGS.config_list_derive: ## all configs in list are full
        return [validate_func(el) for el in cfg_json]

    first_config = validate_func(cfg_json[0])
    first_data = first_config.model_dump()
    return [first_config] + [validate_func(recursive_union(first_data, el)) for el in cfg_json[1:]]


VALID_URLS = {
    CaptureMode.YOUTUBE: ["https://www.youtube.com/", "https://youtu.be/", "https://youtube.com/"],
    CaptureMode.TWITCH: ["https://www.twitch.tv/"],
}


def validate_bounding_box(v: Any) -> Any:
    if v is None:
        return v
    if not isinstance(v, list) and not isinstance(v, tuple):
        raise ValueError("must be of type list or tuple")
    if len(v) != 4:
        raise ValueError("must be of length=4")
    if not all([isinstance(el, int) and el >= 0 for el in v]):
        raise ValueError("all elements must be positive integers")
    return v


InBBox_t: TypeAlias = Annotated[BBox_t, BeforeValidator(validate_bounding_box)]
