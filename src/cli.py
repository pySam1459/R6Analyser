import argparse
from datetime import datetime
from json import load as json_load
from os.path import exists, join
from sys import exit
from time import sleep
from enum import Enum
from typing import TypeVar, Type, Any

from ignmatrix import IGNMatrixMode
from writer import SUPPORTED_SAVEFILE_EXTS
from utils import SaveFile


__all__ = ["main"]


## ----- HELPER FUNCTIONS -----
## Config Parser helper function
def config_arg_error(name: str, reason: str) -> None:
    print(f"CONFIG ERROR: Invalid `{name}` argument, {reason}")
    exit()


def __infer_key(config: dict, key: str) -> None:
    match key:
        case "MAX_ROUNDS":
            config["MAX_ROUNDS"] = 12 if config["SCRIM"] else 15
        case "ROUNDS_PER_SIDE":
            if config["SCRIM"]: config["ROUNDS_PER_SIDE"] = config["MAX_ROUNDS"] // 2
            else: config["ROUNDS_PER_SIDE"] = int((config["MAX_ROUNDS"]-3) // 2)
        case "IGN_MODE":
            config["IGN_MODE"] = IGNMatrixMode.FIXED if len(config["IGNS"]) == 10 else IGNMatrixMode.INFER
        case "TEAM1_SCORE_REGION":
            tr = config["TIMER_REGION"]
            config["TEAM1_SCORE_REGION"] = [tr[0] - tr[2]//2, tr[1], tr[2]//2, tr[3]]
        case "TEAM2_SCORE_REGION":
            tr = config["TIMER_REGION"]
            config["TEAM2_SCORE_REGION"] = [tr[0] + tr[2], tr[1], tr[2]//2, tr[3]]
        case "TEAM1_SIDE_REGION":
            tr = config["TIMER_REGION"]
            config["TEAM1_SIDE_REGION"]  = [tr[0] - int(tr[2]*0.95), tr[1], tr[2]//2, tr[3]]
        case "TEAM2_SIDE_REGION":
            tr = config["TIMER_REGION"]
            config["TEAM2_SIDE_REGION"]  = [tr[0] + int(tr[2]*1.45), tr[1], tr[2]//2, tr[3]]
        case _:
            ...

## The following _cparse_ functions parse and validate different configuration arguments
def __cparse_bool(arg: Any, name: str) -> bool:
    if type(arg) != bool:
        config_arg_error(name, "only Boolean true/false, (no quotation \"\\' marks)")
    return arg

T = TypeVar("T")
def __cparse_type_range(arg: Any, _type: T, name: str, lower: T, upper: T) -> T:
    if type(arg) != _type:
        config_arg_error(name, f"invalid type, arg must be of type {_type}, but got {type(arg)}")
    if not (lower <= arg <= upper):
        config_arg_error(name, f"must be in range [{lower}-{upper}]")
    return arg

def __cparse_bounding_box(arg: Any, name: str) -> list[int]:
    if type(arg) != list or len(arg) != 4 or not all([type(el) == int for el in arg]):
        config_arg_error(name, "must be of length=4 and type list[int]")
    if any([el < 0 for el in arg]):
        config_arg_error(name, "elements must be positive integers")    
    return arg

E = TypeVar('E', bound=Enum)
def __cparse_enum(arg: Any, name: str, enum: Type[E]) -> E: # type: ignore the line
    if type(arg) == enum: return arg

    for enum_member in enum:
        if enum_member.value == arg:
            return enum_member
    config_arg_error(name, f"{arg} not a valid enum value")

def __cparse_IGNS(arg: Any) -> list[str]:
    if type(arg) != list:
        config_arg_error("IGNS", "not a list")
    if not 0 <= len(arg) <= 10:
        config_arg_error("IGNS", "list must have a length of [0-10]")
    if not all([type(el) == str for el in arg]):
        config_arg_error("IGNS", "not a list of strings")
    if not all([1 < len(el) <= 18 for el in arg]): ## max length of an ign is 18?
        config_arg_error("IGNS", "list contains an invalid IGN")
    return arg


## config keys
#    note: MAX_ROUNDS must be inferred before ROUNDS_PER_SIDE
REQUIRED_CONFIG_KEYS = ["SCRIM", "SPECTATOR", "TIMER_REGION", "KF_LINE_REGION", "IGNS"]
INFER_CONFIG_KEYS    = ["MAX_ROUNDS", "ROUNDS_PER_SIDE", "IGN_MODE",
                        "TEAM1_SCORE_REGION", "TEAM2_SCORE_REGION", "TEAM1_SIDE_REGION", "TEAM2_SIDE_REGION"]
OPTIONAL_CONFIG_KEYS = ["SCREENSHOT_RESIZE", "SCREENSHOT_PERIOD"]
DEFAULT_CONFIG_FILENAME = "defaults.json"

DEBUG_KEYS = ["config_keys", "red_percentage"]
DEBUG_FILENAME = "debug.json"

## config parse function for each configuration variable
__cparse_functions = {
    ## Required keys
    "SCRIM":              lambda arg: __cparse_bool(arg, "SCRIM"),
    "SPECTATOR":          lambda arg: __cparse_bool(arg, "SPECTATOR"),
    "TIMER_REGION":       lambda arg: __cparse_bounding_box(arg, "TIMER_REGION"),
    "KF_LINE_REGION":     lambda arg: __cparse_bounding_box(arg, "KF_LINE_REGION"),
    "IGNS":               lambda arg: __cparse_IGNS(arg),

    ## Inferred keys
    "MAX_ROUNDS":         lambda arg: __cparse_type_range(arg, int, "MAX_ROUNDS", 1, 15),
    "ROUNDS_PER_SIDE":    lambda arg: __cparse_type_range(arg, int, "ROUNDS_PER_SIDE", 1, 6),
    "IGN_MODE":           lambda arg: __cparse_enum(arg, "IGN_MODE", IGNMatrixMode),
    "TEAM1_SCORE_REGION": lambda arg: __cparse_bounding_box(arg, "TEAM1_SCORE_REGION"),
    "TEAM2_SCORE_REGION": lambda arg: __cparse_bounding_box(arg, "TEAM2_SCORE_REGION"),
    "TEAM1_SIDE_REGION":  lambda arg: __cparse_bounding_box(arg, "TEAM1_SIDE_REGION"),
    "TEAM2_SIDE_REGION":  lambda arg: __cparse_bounding_box(arg, "TEAM2_SIDE_REGION"),

    ## Optional keys
    "SCREENSHOT_RESIZE":  lambda arg: __cparse_type_range(arg, int, "SCREENSHOT_RESIZE", 1, 8),
    "SCREENSHOT_PERIOD":  lambda arg: __cparse_type_range(arg, float, "SCREENSHOT_PERIOD", 0.25, 2),
}

def __parse_config(config_filepath: str) -> dict:
    ## argument checks
    if not (exists(config_filepath) or exists(join("configs", config_filepath))):
        exit(f"CONFIG ERROR: File '{config_filepath}' cannot be found!")
    
    if not exists(config_filepath):
        config_filepath = join("configs", config_filepath)

    ## Load config from json
    try:
        with open(config_filepath, "r", encoding="utf-8") as f_in:
            config: dict = json_load(f_in)

    except Exception as e:
        exit(f"CONFIG ERROR: Could not open {config_filepath}!\n{str(e)}")

    ## check if all required keys are contained in config file
    for key in REQUIRED_CONFIG_KEYS:
        if key not in config:
            exit(f"CONFIG ERROR: Config file does not contain key '{key}'!")
    
    parsed_keys = []
    for key in config.keys():
        if key not in __cparse_functions:
            exit(f"CONFIG ERROR: Config file contains an invalid key '{key}'!")

        config[key] = __cparse_functions[key](config[key])
        parsed_keys.append(key)
        
    print(f"Info: Loaded configuration file '{config_filepath}'")

    ## Infer keys not present in config file
    keys_to_add = [key for key in INFER_CONFIG_KEYS if key not in config]
    if len(keys_to_add) > 0:
        for key in keys_to_add:
            __infer_key(config, key)

    ## if the optional keys weren't provided, use defaults from default.json
    keys_to_add = [key for key in OPTIONAL_CONFIG_KEYS if key not in config]
    if len(keys_to_add) > 0:
        if not exists(DEFAULT_CONFIG_FILENAME):
            raise FileNotFoundError(f"'{DEFAULT_CONFIG_FILENAME}' does not exist!")

        try:
            with open(DEFAULT_CONFIG_FILENAME, "r", encoding="utf-8") as f_in:
                default_config = json_load(f_in)
        except Exception as e:
            exit(f"CONFIG ERROR: Could not open {DEFAULT_CONFIG_FILENAME}\n{str(e)}")

        for key in keys_to_add:
            if key not in default_config:
                raise Exception(f"defaults.json has been modified, key '{key}' has been removed!")

            config[key] = default_config[key]

    ## final checks, in-case default.json was tampered with
    for key in __cparse_functions.keys() - parsed_keys:
        config[key] = __cparse_functions[key](config[key])

    return config


def __load_debug(debug_filepath: str) -> dict:
    try:
        with open(debug_filepath, "r", encoding="utf-8") as f_in:
            dconfig: dict = json_load(f_in)

    except Exception as e:
        exit(f"CONFIG ERROR: Could not open {debug_filepath}!\n{str(e)}")

    for key in DEBUG_KEYS:
        if key not in dconfig:
            raise Exception(f"debug.json has been modified, key '{key}' has been removed!")
        elif type(dconfig[key]) != bool:
            raise Exception(f"debug/{key} is not boolean type!")
    
    return dconfig


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

def __default_json_savefile() -> str:
    return datetime.now().strftime('%Y-%m-%d_%H-%M-%S') + ".json"


def main():
    parser = argparse.ArgumentParser(
        prog="R6 Analyser",
        description="A Rainbow Six Siege VOD Analyser to record live information from a game.")

    parser.add_argument("-c", "--config",
                        type=__parse_config,
                        help="Filename of the .json config file containing information bounding boxes",
                        dest="config")
    parser.add_argument("-d", "--delay",
                        type=int,
                        help="Time delay between starting the program and recording",
                        dest="delay",
                        default=0)
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
    
    parser.add_argument("--region-tool",
                        action="store_true",
                        help="Runs the Region tool instead of R6Analyser.")
    parser.add_argument("--display",
                        type=int,
                        help="When using the `--region-tool`, Which display to capture",
                        default=0)

    args = parser.parse_args()
    args.debug = __load_debug(DEBUG_FILENAME)

    if args.delay > 0:
        sleep(args.delay)

    if args.region_tool:
        from tools.regiontool import RegionTool

        print("Info: Running Region Tool...")
        RegionTool(args).run()
    
    elif getattr(args, "config", False):
        if args.config["SPECTATOR"]:
            from analyser import SpectatorAnalyser

            print("Info: In Spectator Mode")
            analyser = SpectatorAnalyser(args)
        else:
            from analyser import InPersonAnalyser

            print("Info: In Person Mode")
            analyser = InPersonAnalyser(args)

        if args.test:
            analyser.test()
        else:
            analyser.run()

    else:
        print("Invalid R6Analyser Arguments")


if __name__ == "__main__":
    print("Please run R6Analyser from run.py")
