from pathlib import Path
from sys import exit
from time import sleep
from typing import cast

from analyser import InPersonAnalyser, SpectatorAnalyser
from config import Config, RTConfig, create_config, create_config_list
from settings import create_settings
from tools import RegionTool, check_dependencies
from utils import load_json
from utils.cli import AnalyserArgs
from utils.constants import SETTINGS_PATH


class Controller:
    def __init__(self, args: AnalyserArgs) -> None:
        self.args = args

        if self.args.deps_check:
            check_dependencies()
            exit()

        assert args.config_path is not None, "Config Path is not provided!"

        settings = create_settings(SETTINGS_PATH)
        self.num_games = self.__get_num_games(args.config_path)

        if self.num_games == 1:
            self.config_list = [create_config(args, args.config_path, settings)]
        elif self.num_games > 1:
            self.config_list = create_config_list(args, args.config_path, settings)
        else:
            raise ValueError(f"No configuration provided in {args.config_path}")
    
    def __get_num_games(self, cfg_path: Path) -> int:
        cfg_json = load_json(cfg_path)
        if isinstance(cfg_json, dict):
            return 1
        elif isinstance(cfg_json, list):
            if len(cfg_json) > 0:
                return len(cfg_json)
            else:
                raise ValueError(f"Invalid configuration file, no configs provided! {cfg_path}")
        else:
            raise ValueError(f"Invalid configuration file {cfg_path}")


    def run(self) -> None:
        for i, config in enumerate(self.config_list):
            if self.args.region_tool:
                self.__run_region_tool(cast(RTConfig, config))
            else:
                self.__run_analyser(cast(Config, config))

    def __run_region_tool(self, config: RTConfig) -> None:
        if self.args.delay > 0:
            sleep(self.args.delay)

        print("Info: Running Region Tool...")
        RegionTool.new(self.args, config).run()
    
    def __run_analyser(self, config: Config) -> None:
        if config.spectator:
            print("Info: Spectator Mode Active")
            analyser = SpectatorAnalyser(self.args, config)
        else:
            print("Info: Person Mode Active")
            analyser = InPersonAnalyser(self.args, config)

        if not analyser.is_a_test:
            analyser.run()
