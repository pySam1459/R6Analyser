import argparse
from datetime import datetime
from functools import partial
from json import load as _json_load
from os.path import exists, join
from sys import exit
from time import sleep
from enum import Enum
from typing import TypeAlias, TypeVar, Type, Any, Callable, Optional

from capture import CaptureMode
from ignmatrix import IGNMatrixMode
from writer import SUPPORTED_SAVEFILE_EXTS
from utils import SaveFile, Config


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


def ASSERT(condition: bool, reason: str) -> None:
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

def __partial_type_range(_type: T, lower: T, upper: T) -> partial:
    return partial(__cparse_type_range, _type=_type, lower=lower, upper=upper)

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

def __cparse_IGNS(arg: Any, name: str) -> list[str]:
    if type(arg) != list:
        config_arg_error(name, "not a list")
    if not 0 <= len(arg) <= 10:
        config_arg_error(name, "list must have a length of [0-10]")
    if not all([type(el) == str for el in arg]):
        config_arg_error(name, "not a list of strings")
    if not all([1 < len(el) <= 18 for el in arg]): ## max length of an ign is 18?
        config_arg_error(name, "list contains an invalid IGN")
    return arg

def __cparse_file(arg: Any, name: str) -> str:
    if type(arg) != str:
        config_arg_error(name, "not a string")
    if not exists(arg):
        config_arg_error(name, "file does not exist!")
    return arg


## ----- CONFIGURATION -----
DEBUG_KEYS = ["config", "red_percentage"]
DEBUG_FILENAME = "debug.json"
DEFAULT_CONFIG_FILENAME = "defaults.json"

## <OPTION> values
REQUIRED, DEPENDENT, INFERRED, OPTIONAL = 0x1, 0x2, 0x4, 0x8

"""
Config Specification `spec` is formatted as such:
- leaf key     has a value: tuple[<OPTION>, __cparse_function]
- internal key has a 'sub-config' dictionary value,
    sub-configs follow the same spec format
    sub-configs must have an additional `"<|OPTION|>": <OPTION>` property

The parse functions are implemented above: __cparse_{bool, type_range, bounding_box, enum, IGNS}
  During evaluation, (arg, name) are automatically passed to function
  therefore, use partial functions in specification for other arguments
```
spec_format = {
    key:              (<OPTION>, parse_func),
    key2: {
        "<|OPTION|>":  <OPTION>,
        "key3:        (<OPTION>, parse_func),
        "key4":       { ... }
    }
}
```

Specifications can have dependencies
  - keys which have the DEPENDENT <OPTION>
  - keys that infer on other keys must be inferred later

A config dependency contains the conditions dependencies
```
__deps = {
    "X": ["<|EXIST_IF|>Y<|=|>EXAMPLE VALUE"]
}
```
The Specification is evaluated in the order it is defined
  so inferred keys which depend on other keys must defined after
  all <OPTION>=INFERRED  keys will be inferred after {REQUIRED,OPTIONAL} keys irrespective of definition-order
  all <OPTION>=DEPENDENT keys will be checked last
"""
__spec_t: TypeAlias = dict[str, tuple[int, Callable]|dict]
__config_spec: __spec_t = {
    ## Required keys
    "SCRIM":               (REQUIRED, __cparse_bool),
    "SPECTATOR":           (REQUIRED, __cparse_bool),
    "CAPTURE": {
        "<|OPTION|>":       REQUIRED,
        "MODE":            (REQUIRED, partial(__cparse_enum, enum=CaptureMode)),
        "FILE":            (DEPENDENT, __cparse_file),
        "REGIONS": {
            "<|OPTION|>":   REQUIRED,
            "TIMER":       (REQUIRED, __cparse_bounding_box),
            "KF_LINE":     (REQUIRED, __cparse_bounding_box),
            # Inferred Keys
            "TEAM1_SCORE": (INFERRED, __cparse_bounding_box),
            "TEAM2_SCORE": (INFERRED, __cparse_bounding_box),
            "TEAM1_SIDE":  (INFERRED, __cparse_bounding_box),
            "TEAM2_SIDE":  (INFERRED, __cparse_bounding_box)
        }
    },
    "IGNS":                (REQUIRED, __cparse_IGNS),

    ## Inferred keys
    "IGN_MODE":            (INFERRED, partial(__cparse_enum, enum=IGNMatrixMode)),
    "MAX_ROUNDS":          (INFERRED, __partial_type_range(int, 1, 15)),
    "ROUNDS_PER_SIDE":     (INFERRED, __partial_type_range(int, 1, 6)),

    ## Optional keys
    "SCREENSHOT_RESIZE":   (OPTIONAL, __partial_type_range(int, 1, 8)),
    "SCREENSHOT_PERIOD":   (OPTIONAL, __partial_type_range(float, 0.0, 2.0)),
}

## dependencies: dict[key: required if everything here == true]
__deps_t: TypeAlias = dict[str, list[str]]
__config_deps = {
    "CAPTURE": {
        "FILE": [lambda cfg: cfg.capture.mode != CaptureMode.SCREENSHOT],
    }
}

__regiontool_spec: __spec_t = {
    "SPECTATOR":           (REQUIRED, __cparse_bool),
    "CAPTURE": {
        "<|OPTION|>":       REQUIRED,
        "MODE":            (REQUIRED, partial(__cparse_enum, enum=CaptureMode)),
        "FILE":            (DEPENDENT, __cparse_file),
        "REGIONS": {
            "<|OPTION|>":   REQUIRED,
        }
    },
}

__regiontool_deps = {
    "CAPTURE": {
        "FILE": [lambda cfg: cfg.capture.mode != CaptureMode.SCREENSHOT],
    }
}


def __parse_config(config_filepath: str, args: argparse.Namespace) -> Config | list[Config]:
    ## argument checks
    if not (exists(config_filepath) or exists(join("configs", config_filepath))):
        exit(f"CONFIG ERROR: File '{config_filepath}' cannot be found!")

    if not exists(config_filepath):
        config_filepath = join("configs", config_filepath)

    ## Load config,defaults from json
    config = load_json(config_filepath)
    default_cfg = Config(load_json(DEFAULT_CONFIG_FILENAME), name="defaults")

    spec = __config_spec
    deps = __config_deps
    if args.region_tool:
        spec = __regiontool_spec
        deps = __regiontool_deps

    if type(config) == list:
        if len(config) == 1:
            config = config[0]
        elif len(config) > 1:
            config[0]["CFG_FILE_PATH"] = config_filepath
            cfg0 = __validate_config(Config(config[0], name="config0"), default_cfg, spec, deps)
            other_cfgs = [__infer_config(Config(cfg, name=f"config{i}"), cfg0) for i, cfg in enumerate(config[1:], start=1)]
            return [cfg0] + other_cfgs
        else:
            raise TypeError("CONFIG ERROR: Config list is empty")

    if type(config) == dict:
        ## Pass config,spec,defaults to recursive validator function
        config["CFG_FILE_PATH"] = config_filepath
        return __validate_config(Config(config, name="config"), default_cfg, spec, deps)

    else:
        raise TypeError("CONFIG ERROR: Invalid config root type")


def __validate_config(config: Config, default_cfg: Config,
                      spec: __spec_t, deps: __deps_t,
                      key_path: str = "",
                      infer_list: Optional[list[tuple]] = None, dep_list: Optional[list[tuple]] = None) -> Config:
    """Validates and parses the structure and values in the config given the spec"""
    __first = False
    if infer_list is None:
        infer_list = []
        dep_list = []
        __first = True

    for skey, svalue in spec.items():
        nkey_path = f"{key_path}/{skey}".strip(r"/") ## eg. IGNS, CAPTURE/MODE, CAPTURE/REGIONS/TIMER

        if skey == "<|OPTION|>": continue
        if skey in config:
            if type(config[skey]) == Config:
                ASSERT(type(svalue) == dict, f"CONFIG ERROR: {nkey_path} must be a dict not value")
                config[skey] = __validate_config(config[skey], default_cfg.get(skey, {}), svalue, deps.get(skey, {}),
                                                 nkey_path, infer_list, dep_list)
            else:
                ASSERT(type(svalue) != dict, f"CONFIG ERROR: {nkey_path} must be a value not dict")
                ASSERT(len(svalue) == 2, f"SPEC ERROR: {nkey_path} is not valid (<OPTION>, __parse_func)")
                config[skey] = svalue[1](arg=config[skey], name=nkey_path)

        else:
            if type(svalue) == dict:
                ASSERT("<|OPTION|>" in svalue, f"SPEC ERROR: The config spec has not set the <|OPTION|> for a sub-config")
                opt = svalue["<|OPTION|>"]
            else:
                opt = svalue[0]

            if opt & REQUIRED:
                exit(f"CONFIG ERROR: Config file does not contain required key '{nkey_path}'!")
            if opt & DEPENDENT:
                dep_list.append((nkey_path, skey, deps))
            if opt & OPTIONAL:
                ASSERT(skey in default_cfg, f"{nkey_path} is missing from default.json!")
                config[skey] = default_cfg[skey]
            if opt & INFERRED:
                infer_list.append((nkey_path, skey, config))

    if not __first:
        return config

    ## infer properties after required,optional key validations
    for key_path, key_last, cfg_parent in infer_list:
        cfg_parent[key_last] = __infer_from_key(config, key_path)

    ## check dependencies of DEPENDENT properties last
    for key_path, key_last, dep_parent in dep_list:
        ASSERT(not all([func(config) for func in dep_parent[key_last]]),
               f"CONFIG ERROR: Dependent property {key_path}'s condition is not met.")

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


def __infer_config(config: Config, src_cfg: Config) -> None:
    return config


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

    parser.add_argument("config",
                        help="JSON configuration file located in . or ./configs")
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
    args.debug = __load_debug(DEBUG_FILENAME)
    args.config = __parse_config(args.config, args)

    if args.delay > 0:
        sleep(args.delay)

    if args.region_tool:
        from tools.regiontool import RegionTool

        print("Info: Running Region Tool...")
        rt = RegionTool.new(args)
        rt.run()
    
    elif getattr(args, "config", False):
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
