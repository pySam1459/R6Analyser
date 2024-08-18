import argparse
import requests as r
from datetime import datetime
from re import fullmatch
from sys import exit
from time import sleep
from dotenv import dotenv_values
from typing import Optional

from writer import SUPPORTED_SAVEFILE_EXTS
from config import Config, create_config
from utils import SaveFile, load_json


__all__ = [ "main" ]


DEBUG_FILENAME = "debug.json"
DEBUG_KEYS = ["config_keys", "red_percentage"]


def __load_debug(debug_filepath: str) -> Config:
    dconfig = load_json(debug_filepath)

    for key in DEBUG_KEYS:
        if key not in dconfig:
            raise KeyError(f"debug.json has been modified, key '{key}' has been removed!")
        elif not isinstance(dconfig[key], bool):
            raise ValueError(f"debug/{key} is not boolean type!")
    
    return Config(dconfig)


def __parse_key(arg: str) -> str:
    if fullmatch("^[a-f0-9]{64}$", arg.strip()):
        return arg
    
    raise argparse.ArgumentTypeError("Invalid software key! Must be a 32-length hex string")


def __parse_verbose(arg: str) -> int:
    try:
        x = int(arg)
        if 0 <= x <= 3:
            return x
        raise argparse.ArgumentTypeError("Verbose argument out of range [0,3]")

    except ValueError:
        raise argparse.ArgumentTypeError(f"Invalid Verbose argument {arg}")


def __parse_save(arg: str) -> SaveFile:
    if "." not in arg:
        raise argparse.ArgumentTypeError(f"Invalid save file {arg}")

    filename, ext = arg.rsplit(".", maxsplit=1)
    if ext in SUPPORTED_SAVEFILE_EXTS:
        return SaveFile(filename, ext)
    raise argparse.ArgumentTypeError(f"Invalid save file type {ext}, only json/xlsx allowed")


def __choose_secret_key(env_key: Optional[str], arg_key: Optional[str]) -> Optional[str]:
    if env_key is None and arg_key is None:
        return None
    if arg_key is not None:
        return arg_key
    return env_key


def __validate_secret_key(secret_key: str) -> Optional[dict]:
    data = {"key": secret_key}
    headers = {"Content-Type": "application/json"}
    res = r.post("https://sambar6analyser.com/keycheck", json=data, headers=headers)

    res_data: dict = res.json()
    if res.status_code == 200 and res_data.get("status", None) == "success":
        print("Valid Secret Key")
        return res_data
    
    error_res = res.json().get("detail")
    print(f"Secret Key validation error code {res.status_code}: {error_res}")
    return None


def validate_secret_key(arg_key: str):
    env = dotenv_values()
    secret_key = __choose_secret_key(env.get("SOFTWARE-KEY", None), arg_key)
    if not secret_key:
        print("No software key provided, please either add `SECRET-KEY=` to .env or use --key cli argument")
        exit()
    if not __validate_secret_key(secret_key):
        exit()


def __default_json_savefile() -> str:
    return datetime.now().strftime('%Y-%m-%d_%H-%M-%S') + ".json"


def main():
    parser = argparse.ArgumentParser(
        prog="R6 Analyser",
        description="A Rainbow Six Siege VOD Analyser to record live information from a game.")

    parser.add_argument("config",
                        help="JSON configuration file located in . or ./configs")
    parser.add_argument("-k", "--key",
                        type=__parse_key,
                        help="Software Key",
                        dest="key",
                        default=None)
    parser.add_argument("-v", "--verbose",
                        type=__parse_verbose,
                        help="Determines how detailed the console output is, 0-nothing, 1-some, 2-all, 3-debug",
                        dest="verbose",
                        default=2)
    parser.add_argument("-s", "--save",
                        type=__parse_save,
                        help="Where to save the output of R6Analyser, only json or xlsx files",
                        dest="save",
                        default=__default_json_savefile())
    parser.add_argument("--append-save",
                       action="store_true",
                       help="Whether to append new round data onto the existing save file, otherwise save all data at the end of a game",
                       dest="append_save")
    # parser.add_argument("--upload-save",
    #                    action="store_true",
    #                    help="Whether to upload new round data directly to the cloud, otherwise save all data at the end of a game",
    #                    dest="upload_save")
    parser.add_argument("--check", 
                        action="store_true",
                        help="Does not perform data extract but saves the regions of interest as images for quality check")
    parser.add_argument("--test",
                        action="store_true",
                        help="Performs data extract for the instance when the program is run")
    parser.add_argument("--cpu",
                        action="store_true",
                        help="Flag for only cpu execution, if your machine does not support gpu acceleration")

    parser.add_argument("-d", "--delay",
                        type=int,
                        help="Time delay between starting the program and recording",
                        dest="delay",
                        default=0)

    parser.add_argument("--region-tool",
                        action="store_true",
                        help="Runs the Region tool instead of R6Analyser.")
    parser.add_argument("--display",
                        type=int,
                        help="When using the `--region-tool`, Which display to capture",
                        default=0)

    args = parser.parse_args()
    
    if not (args.test or args.check or args.region_tool):
        validate_secret_key(args.key)

    args.debug = __load_debug(DEBUG_FILENAME)
    args.config = create_config(args.config, args)

    if args.delay > 0:
        sleep(args.delay)

    if args.region_tool:
        from tools.regiontool import RegionTool

        print("Info: Running Region Tool...")
        rt = RegionTool.new(args)
        rt.run()
    
    elif getattr(args, "config", False):
        assert isinstance(args.config, Config), "List Configs are not supported yet"
        if args.config.spectator:
            from analyser import SpectatorAnalyser

            print("Info: Spectator Mode Active")
            analyser = SpectatorAnalyser(args)
        else:
            from analyser import InPersonAnalyser

            print("Info: Person Mode Active")
            analyser = InPersonAnalyser(args)

        if args.test:
            analyser.test()
        else:
            analyser.run()

    else:
        print("Invalid R6Analyser Arguments")


if __name__ == "__main__":
    print("Please run R6Analyser from run.py")
