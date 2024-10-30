from pathlib import Path
from typing import cast

from args import AnalyserArgs
from settings import Settings

from .analyser_cfg import Config, create_analyser_config
from .regiontool_cfg import RTConfig, RTRegionsCFG, create_regiontool_config


__all__ = [
    "create_config",
    "create_config_list",
    "Config",
    "RTConfig",
    "RTRegionsCFG"
]


def create_config(args: AnalyserArgs,
                  config_path: Path,
                  settings: Settings) -> Config | RTConfig:
    if args.region_tool:
        return cast(RTConfig, create_regiontool_config(config_path, settings))
    else:
        return cast(Config, create_analyser_config(config_path, settings))


def create_config_list(args: AnalyserArgs,
                       config_path: Path,
                       settings: Settings) -> list[Config] | list[RTConfig]:
    if args.region_tool:
        return cast(list[RTConfig], create_regiontool_config(config_path, settings))
    else:
        return cast(list[Config], create_analyser_config(config_path, settings))
