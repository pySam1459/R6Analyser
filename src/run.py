import argparse
from json import load as json_load
from os.path import exists, join
from sys import exit
from time import sleep
from datetime import datetime
from enum import Enum
from typing import TypeVar, Type
from analyser import InPersonAnalyser, SpectatorAnalyser, SaveFile, IGNMatrixMode


## ----- HELPER FUNCTIONS -----
def __infer_key(config: dict, key: str) -> None:
    match key:
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
        case "MAX_ROUNDS":
            config["MAX_ROUNDS"] = 12 if config["SCRIM"] else 15
        case "ROUNDS_PER_SIDE":
            config["ROUNDS_PER_SIDE"] = (config["MAX_ROUNDS"]-3)/2
        case _:
            ...

## Config Parser helper function
def arg_error(name: str, reason: str) -> argparse.ArgumentError:
    print(f"CONFIG ERROR: Invalid `{name}` argument, {reason}")
    exit()

def __cparse_bool(arg, name: str) -> bool:
    if type(arg) != bool:
        arg_error(name, "only Boolean (true/false)")
    return arg

T = TypeVar("T")
def __cparse_type_range(arg, _type: T|list[T], name: str, lower: T, upper: T) -> T:
    if type(_type) != list:
        _type = [_type]
    if not any([type(arg) == t for t in _type]):
        s_type = ",".join([str(t) for t in _type])
        arg_error(name, f"only {s_type} types")
    if not (lower <= arg <= upper):
        arg_error(name, f"must be in range [{lower}-{upper}]")
    return arg

def __cparse_bounding_box(arg, name: str) -> list[int]:
    if type(arg) != list or len(arg) != 4 or not all([type(el) == int for el in arg]):
        arg_error(name, "must be of length=4 and type list[int]")
    if any([el < 0 for el in arg]):
        arg_error(name, "elements must be positive integers")    
    return arg

E = TypeVar('E', bound=Enum)
def __cparse_enum(arg, name: str, enum: Type[E]) -> E:
    for enum_member in enum:
        if enum_member.value == arg:
            return enum_member
    arg_error(name, f"{arg} not a valid enum value")

def __cparse_IGNS(arg) -> list[str]:
    if type(arg) != list:
        arg_error("IGNS", "not a list")
    if not 0 <= len(arg) <= 10:
        arg_error("IGNS", "list must have a length of [0-10]")
    if not all([type(el) == str for el in arg]):
        arg_error("IGNS", "not a list of strings")
    
    return arg


## config keys
#    note: MAX_ROUNDS must be inferred before ROUNDS_PER_SIDE
REQUIRED_CONFIG_KEYS = ["SCRIM", "TIMER_REGION", "KILLFEED_REGION"]
INFER_CONFIG_KEYS    = ["TEAM1_SCORE_REGION", "TEAM2_SCORE_REGION", "TEAM1_SIDE_REGION", "TEAM2_SIDE_REGION", "MAX_ROUNDS", "ROUNDS_PER_SIDE"]
OPTIONAL_CONFIG_KEYS = ["SCREENSHOT_RESIZE", "SCREENSHOT_PERIOD", "IGNS", "IGN_MODE", "SPECTATOR"]
DEFAULT_CONFIG_FILENAME = "defaults.json"

## config parse function for each configuration variable
__cparse_functions = {
    "TIMER_REGION":       lambda arg: __cparse_bounding_box(arg, "TIMER_REGION"),
    "KILLFEED_REGION":    lambda arg: __cparse_bounding_box(arg, "KILLFEED_REGION"),
    "SCRIM":              lambda arg: __cparse_bool(arg, "SCRIM"),
    "MAX_ROUNDS":         lambda arg: __cparse_type_range(arg, int, "MAX_ROUNDS", 1, 15),
    "ROUNDS_PER_SIDE":    lambda arg: __cparse_type_range(arg, int, "ROUNDS_PER_SIDE", 1, 6),
    "SPECTATOR":          lambda arg: __cparse_bool(arg, "SPECTATOR"),
    "IGNS":               lambda arg: __cparse_IGNS(arg),
    "IGN_MODE":           lambda arg: __cparse_enum(arg, "IGN_MODE", IGNMatrixMode),
    "SCREENSHOT_RESIZE":  lambda arg: __cparse_type_range(arg, [int, float], "SCREENSHOT_RESIZE", 1, 8),
    "SCREENSHOT_PERIOD":  lambda arg: __cparse_type_range(arg, [int, float], "SCREENSHOT_PERIOD", 0.25, 2),
    "TEAM1_SCORE_REGION": lambda arg: __cparse_bounding_box(arg, "TEAM1_SCORE_REGION"),
    "TEAM2_SCORE_REGION": lambda arg: __cparse_bounding_box(arg, "TEAM2_SCORE_REGION"),
    "TEAM1_SIDE_REGION":  lambda arg: __cparse_bounding_box(arg, "TEAM1_SIDE_REGION"),
    "TEAM2_SIDE_REGION":  lambda arg: __cparse_bounding_box(arg, "TEAM2_SIDE_REGION"),
}

def __parse_config(arg: str) -> dict:
    ## argument checks
    if not (exists(arg) or exists(join("configs", arg))):
        raise argparse.ArgumentError(f"Config file '{arg}' cannot be found!")
    
    if not exists(arg):
        arg = join("configs", arg)

    ## Load config from json
    with open(arg, "r", encoding="utf-8") as f_in:
        config: dict = json_load(f_in)

    ## check if all required keys are contained in config file
    for key in REQUIRED_CONFIG_KEYS:
        if key not in config:
            raise argparse.ArgumentError(f"Config file does not contain key '{key}'!")
    
    validated_keys = []
    for key in config.keys():
        config[key] = __cparse_functions[key](config[key])
        validated_keys.append(key)
        
    print(f"Info: Loaded configuration file '{arg}'") 
    to_add = [key for key in INFER_CONFIG_KEYS if key not in config]
    if len(to_add) > 0:
        for key in to_add:
            __infer_key(config, key)

    ## if the optional keys weren't provided, use defaults from default.json
    to_add = [key for key in OPTIONAL_CONFIG_KEYS if key not in config]
    if len(to_add) > 0:
        if not exists(DEFAULT_CONFIG_FILENAME):
            raise argparse.ArgumentError("'defaults.json' does not exist!")

        with open(DEFAULT_CONFIG_FILENAME, "r", encoding="utf-8") as f_in:
            default_config = json_load(f_in)

        __log = ""
        for key in to_add:
            if key not in default_config:
                raise Exception(f"defaults.json has been modified, key '{key}' has been removed!")

            config[key] = default_config[key]
            __log += f"{key}, "
        print(f"Info: Loaded default config keys {__log}")

    ## final checks, in-case default.json was tampered with
    for key, func in __cparse_functions.items():
        if key not in validated_keys:
            config[key] = func(config[key])

    return config


def __parse_verbose(arg: str) -> int:
    try:
        x = int(arg)
        if 0 <= x <= 3:
            return x
        raise argparse.ArgumentError("Verbose argument out of range [0,2]")

    except ValueError:
        raise argparse.ArgumentError(f"Invalid Verbose argument {arg}")


def __parse_save(arg: str) -> SaveFile:
    filename, ext = arg.rsplit(".", maxsplit=1)
    if ext == "json" or ext == "xlsx":
        return SaveFile(filename, ext)
    raise argparse.ArgumentError(f"Invalid save file type {ext}, only json/xlsx allowed")

def __default_save() -> str:
    return datetime.now().strftime('%Y-%m-%d_%H-%M-%S') + ".json"


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        prog="R6 Analyser",
        description="A Rainbow Six Siege VOD Analyser to record live information from a game.")

    parser.add_argument("config",
                        type=__parse_config,
                        help="Filename of the .json config file containing information bounding boxes")
    parser.add_argument("-d", "--delay",
                        type=int,
                        help="Time delay between starting the program and recording",
                        dest="delay",
                        default=0)
    parser.add_argument("-v", "--verbose",
                        type=__parse_verbose,
                        help="Determines how detailed the console output is, 0-nothing, 1-some, 2-all, 3-debug",
                        dest="verbose",
                        default=1)
    parser.add_argument("-s", "--save",
                        type=__parse_save,
                        help="Where to save the output of R6Analyser, only json or xlsx files",
                        dest="save",
                        default=__default_save())
    parser.add_argument("--append-save",
                       action="store_true",
                       help="Whether to append new round data onto the existing save file, otherwise save all data at the end of a game",
                       dest="append_save")
    parser.add_argument("--upload-save",
                       action="store_true",
                       help="Whether to upload new round data directly to the cloud, otherwise save all data at the end of a game",
                       dest="upload_save")
    parser.add_argument("--check", 
                        action="store_true",
                        help="Does not perform data extract but saves the regions of interest as images for quality check")
    parser.add_argument("--test",
                        action="store_true",
                        help="Performs data extract for the instance when the program is run")
    parser.add_argument("--cpu",
                        action="store_true",
                        help="Flag for only cpu execution, if your machine does not support gpu acceleration")

    args = parser.parse_args()
    if args.delay > 0:
        sleep(args.delay)

    if args.config.get("SPECTATOR", False):
        print("Info: In Spectator Mode")
        analyser = SpectatorAnalyser(args)
    else:
        print("Info: In Person Mode")
        analyser = InPersonAnalyser(args)

    if args.test:
        analyser.test()
    else:
        analyser.run()
