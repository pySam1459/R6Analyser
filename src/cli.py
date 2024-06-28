import argparse
from datetime import datetime
from json import load as _json_load
from os.path import exists, join
from sys import exit
from time import sleep
from enum import Enum
from typing import TypeAlias, TypeVar, Type, Any, Callable, Optional

from capture import CaptureMode
from ignmatrix import IGNMatrixMode
from writer import SUPPORTED_SAVEFILE_EXTS
from utils import SaveFile, Config, topological_sort


__all__ = [
    "main",
    "Config"
]


## ----- HELPER FUNCTIONS -----
## Config Parser helper function
def load_json(file_path: str) -> dict:
    """Loads json from `file_path` and handles any errors"""
    if not exists(file_path):
        raise FileNotFoundError(f"'{file_path}' does not exist!")
    
    try:
        with open(file_path, "r", encoding="utf-8") as f_in:
            return _json_load(f_in)

    except Exception as e:
        exit(f"JSON LOAD ERROR: Could not open {file_path}!\n{str(e)}")


def assert_eq(condition: bool, reason: str) -> None:
    if not condition:
        print(reason)
        exit()


def config_arg_error(name: str, reason: str) -> None:
    print(f"CONFIG ERROR: Invalid `{name}` argument, {reason}")
    exit()


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


## ----- CONFIGURATION -----
DEBUG_KEYS = ["config", "red_percentage"]
DEBUG_FILENAME = "debug.json"
DEFAULT_CONFIG_FILENAME = "defaults.json"

## <OPTION> values
REQUIRED, INFERRED, OPTIONAL = 1, 2, 3

"""
Config Specification `spec` is formatted as such:
- leaf key     has a value: tuple[<OPTION>, parse function]
- internal key has a 'sub-config' dictionary value,
    sub-configs follow the same spec format

The parse functions are implemented above:
  __cparse_{bool, type_range, bounding_box, enum, IGNS}

spec format = {
    key: (<OPTION>, parse func),
    key2: {
        "__option": <OPTION>,            ## this key 
        "key3: (<OPTION>, parse func),
        "key4": { ... }
    }
}

For inferred config values, `__infer_deps` specifies which config keys
  must be inferred before others (infer-dependencies)
  eg. keys A,B are both inferred, A is inferred using B, B must be inferred before A

__infer_deps = {
    "B": ["A"],
    "C": ["B", "D"]
}
"""
__spec_t: TypeAlias = dict[str, tuple[int, Callable]|dict]
__config_spec: __spec_t = {
    ## Required keys
    "SCRIM":               (REQUIRED, lambda arg: __cparse_bool(arg, "SCRIM")),
    "SPECTATOR":           (REQUIRED, lambda arg: __cparse_bool(arg, "SPECTATOR")),
    "CAPTURE": {
        "__option":         REQUIRED,
        "MODE":            (REQUIRED, lambda arg: __cparse_enum(arg, "CAPTURE/MODE", CaptureMode)),
        "REGIONS": {
            "__option":     REQUIRED,
            "TIMER":       (REQUIRED, lambda arg: __cparse_bounding_box(arg, "CAPTURE/REGIONS/TIMER")),
            "KF_LINE":     (REQUIRED, lambda arg: __cparse_bounding_box(arg, "CAPTURE/REGIONS/KF_LINE")),
            # Inferred Keys
            "TEAM1_SCORE": (INFERRED, lambda arg: __cparse_bounding_box(arg, "CAPTURE/REGIONS/TEAM1_SCORE")),
            "TEAM2_SCORE": (INFERRED, lambda arg: __cparse_bounding_box(arg, "CAPTURE/REGIONS/TEAM2_SCORE")),
            "TEAM1_SIDE":  (INFERRED, lambda arg: __cparse_bounding_box(arg, "CAPTURE/REGIONS/TEAM1_SIDE")),
            "TEAM2_SIDE":  (INFERRED, lambda arg: __cparse_bounding_box(arg, "CAPTURE/REGIONS/TEAM2_SIDE"))
        }
    },
    "IGNS":                (REQUIRED, lambda arg: __cparse_IGNS(arg)),

    ## Inferred keys
    "MAX_ROUNDS":          (INFERRED, lambda arg: __cparse_type_range(arg, int, "MAX_ROUNDS", 1, 15)),
    "ROUNDS_PER_SIDE":     (INFERRED, lambda arg: __cparse_type_range(arg, int, "ROUNDS_PER_SIDE", 1, 6)),
    "IGN_MODE":            (INFERRED, lambda arg: __cparse_enum(arg, "IGN_MODE", IGNMatrixMode)),

    ## Optional keys
    "SCREENSHOT_RESIZE":   (OPTIONAL, lambda arg: __cparse_type_range(arg, int, "SCREENSHOT_RESIZE", 1, 8)),
    "SCREENSHOT_PERIOD":   (OPTIONAL, lambda arg: __cparse_type_range(arg, float, "SCREENSHOT_PERIOD", 0.0, 2.0)),
}

## infer-dependencies: dict[key: depends on everything here]
__infer_deps = {
    "MAX_ROUNDS": ["ROUNDS_PER_SIDE"]
}


def __parse_config(config_filepath: str) -> Config:
    ## argument checks
    if not (exists(config_filepath) or exists(join("configs", config_filepath))):
        exit(f"CONFIG ERROR: File '{config_filepath}' cannot be found!")

    if not exists(config_filepath):
        config_filepath = join("configs", config_filepath)

    ## Load config,defaults from json
    config = load_json(config_filepath)
    default_cfg = load_json(DEFAULT_CONFIG_FILENAME)

    ## Pass config,spec,defaults to recursive validator funciton
    return __validate_config(Config(config, name="config"), __config_spec, Config(default_cfg, name="defaults"))


def __validate_config(config: Config, spec: __spec_t, default_cfg: Config,
                      key_path: str = "", infer_list: Optional[list[str]] = None) -> Config:
    """Validates and parses the structure and values in the config given the spec"""
    __first = False
    if infer_list is None:
        infer_list = []
        __first = True

    for skey, svalue in spec.items():
        nkey_path = f"{key_path}/{skey}".strip(r"/") ## eg. IGNS, CAPTURE/MODE, CAPTURE/REGIONS/TIMER

        if skey == "__option": continue
        elif skey in config:
            if type(config[skey]) == Config:
                assert_eq(type(config[skey]) == Config, f"Invalid Configuration: {nkey_path} must be a dict not value")
                config[skey] = __validate_config(config[skey], svalue, default_cfg.get(skey, {}), nkey_path, infer_list)
            else:
                assert_eq(type(config[skey]) != Config, f"Invalid Configuration: {nkey_path} must be a value not dict")
                config[skey] = svalue[1](config[skey])

        ## spec key is NOT in the config
        elif type(svalue) == Config:
            assert_eq("__option" in svalue, f"SPEC ERROR: The config spec has not set the __option for a sub-config")
            opt = svalue["__option"]
            if opt == REQUIRED:
                exit(f"CONFIG ERROR: Config file does not contain required key '{nkey_path}'!")
            elif opt == OPTIONAL:
                assert_eq(skey in default_cfg, f"{nkey_path} is missing from default.json!")
                config[skey] = default_cfg[skey]
            elif opt == INFERRED:
                infer_list.append((nkey_path, skey, config))
            else:
                exit(f"SPEC ERROR: The config spec is incorrect, out of valid option {opt}")

        elif svalue[0] == REQUIRED:
            exit(f"CONFIG ERROR: Config file does not contain required key '{nkey_path}'!")

        elif svalue[0] == OPTIONAL:
            assert_eq(skey in default_cfg, f"{nkey_path} is missing from default.json!")
            config[skey] = default_cfg[skey]

        elif svalue[0] == INFERRED:
            infer_list.append((nkey_path, skey, config))

    if not __first:
        return config

    ## infer_list must be sorted due to infer-dependencies
    topological_sort(infer_list, __infer_deps)
    for key_path, key_last, cfg_parent in infer_list:
        cfg_parent[key_last] = __infer_from_key(config, key_path)

    return config


def __infer_from_key(config: Config, key: str) -> Any:
    match key:
        case "MAX_ROUNDS":
            return 12 if config.scrim else 15

        case "ROUNDS_PER_SIDE":
            if config.scrim: return config.max_rounds // 2
            else: return int((config.max_rounds-3) // 2)

        case "IGN_MODE":
            return IGNMatrixMode.FIXED if len(config.igns) >= 10 else IGNMatrixMode.INFER

        case "CAPTURE/REGIONS/TEAM1_SCORE":
            tr = config.capture.regions.timer
            return [tr[0] - tr[2]//2, tr[1], tr[2]//2, tr[3]]

        case "CAPTURE/REGIONS/TEAM2_SCORE":
            tr = config.capture.regions.timer
            return [tr[0] + tr[2], tr[1], tr[2]//2, tr[3]]

        case "CAPTURE/REGIONS/TEAM1_SIDE":
            tr = config.capture.regions.timer
            return [tr[0] - int(tr[2]*0.95), tr[1], tr[2]//2, tr[3]]

        case "CAPTURE/REGIONS/TEAM2_SIDE":
            tr = config.capture.regions.timer
            return [tr[0] + int(tr[2]*1.45), tr[1], tr[2]//2, tr[3]]

        case _:
            return None


def __load_debug(debug_filepath: str) -> Config:
    dconfig = load_json(debug_filepath)

    for key in DEBUG_KEYS:
        if key not in dconfig:
            raise KeyError(f"debug.json has been modified, key '{key}' has been removed!")
        elif type(dconfig[key]) != bool:
            raise ValueError(f"debug/{key} is not boolean type!")
    
    return Config(dconfig)


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
        if args.config.spectator:
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
