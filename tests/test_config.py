import os
import pytest
from pathlib import Path
from re import sub
from pydantic import ValidationError
from unittest.mock import patch
from typing import Any

import utils
from config.analyser_cfg import create_analyser_config
from config.regiontool_cfg import create_regiontool_config
from settings import Settings


def load_configs(filename: str | list[str]) -> list[dict[str, Any]]:
    base = Path(__file__).parent / "resources" / "configs"
    if isinstance(filename, str):
        filename = [filename]
    
    return utils.flatten([utils.load_json(base / file) for file in filename])

def _convert_raw_name(name: str) -> str:
    pattern = r"^(\w+)_(\w+)\.(\d+)-(.+)$"
    replacement = r"\1_\2-\4"
    
    return sub(pattern, replacement, name)

def get_config_name(config: dict[str, Any] | list[dict[str, Any]]) -> str:
    if isinstance(config, list):
        first_name = next((cfg["name"] for cfg in config if cfg.get("name", False)), None)
        if first_name is None:
            return "Test Config"
        return _convert_raw_name(first_name)

    elif isinstance(config, dict):
        return config.get("name", "Test Config")

    raise ValueError(f"Invalid config type {type(config)}")


@pytest.fixture
def cfg_path() -> Path:
    return Path("configs") / "test.json"

@pytest.fixture
def settings() -> Settings:
    return Settings(config_list_derive=True)


## ------ Good Analyser Configs ------
@pytest.fixture(
        params=load_configs(["good_analyser_configs.json", "good_list_analyser_configs.json"]),
        ids=get_config_name)
def cfg_dict_ga(request):
    return request.param

def test_good_analyser_validate(cfg_dict_ga, cfg_path, settings) -> None:
    with (patch.object(Path, "exists", return_value=True),
          patch.object(os,   "access", return_value=True),
          patch.object(os, "makedirs"),
          patch("config.utils.load_json", return_value=cfg_dict_ga)):
        create_analyser_config(cfg_path, settings)


## ------ Good RegionTool Configs ------
@pytest.fixture(
    params=load_configs(["good_regiontool_configs.json", "good_list_regiontool_configs.json"]),
    ids=get_config_name
)
def cfg_dict_grt(request):
    return request.param

def test_good_regiontool_validate(cfg_dict_grt, cfg_path, settings) -> None:
    with (patch.object(Path, "exists", return_value=True),
          patch.object(os, "access", return_value=True),
          patch.object(os, "makedirs"),
          patch("config.utils.load_json", return_value=cfg_dict_grt)):
        create_regiontool_config(cfg_path, settings)


# ## ------ Bad Analyser Configs ------
@pytest.fixture(
    params=load_configs(["bad_analyser_configs.json", "bad_list_analyser_configs.json"]),
    ids=get_config_name
)
def cfg_dict_ba(request):
    return request.param

def test_bad_analyser_validate(cfg_dict_ba, cfg_path, settings) -> None:
    with (pytest.raises(ValidationError),
            patch.object(Path, "exists", return_value=True),
            patch.object(os, "access", return_value=True),
            patch.object(os, "makedirs"),
            patch("config.utils.load_json", return_value=cfg_dict_ba)):
        create_analyser_config(cfg_path, settings)


# ## ------ Bad RegionTool Configs ------
@pytest.fixture(
    params=load_configs("bad_regiontool_configs.json"),
    ids=get_config_name
)
def cfg_dict_brt(request):
    return request.param

def test_bad_regiontool_validate(cfg_dict_brt, cfg_path, settings) -> None:
    with (pytest.raises(ValidationError),
            patch.object(Path, "exists", return_value=True),
            patch.object(os, "access", return_value=True),
            patch.object(os, "makedirs"),
            patch("config.utils.load_json", return_value=cfg_dict_brt)):
        create_regiontool_config(cfg_path, settings)
