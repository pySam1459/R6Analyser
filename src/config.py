import argparse
import json
from enum import Enum
from functools import partial
from os.path import exists, join
from typing import Any, Optional, Type, TypeVar, TypeAlias, Callable

from ignmatrix import IGNMatrixMode
from utils import load_json
from enums import CaptureMode


__all__ = [ "Config", "create_config" ]


def config_arg_error(name: str, reason: str) -> None:
    print(f"CONFIG ERROR: Invalid `{name}` argument, {reason}")
    exit()


def ASSERT(condition: bool, reason: str) -> None:
    if not condition:
        print(reason)
        exit()


## The following _cparse_ functions parse and validate different configuration arguments
def __cparse_bool(arg: Any, name: str) -> bool:
    if not isinstance(arg, bool):
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
    if not isinstance(arg, list) or len(arg) != 4 or not all([type(el) == int for el in arg]):
        config_arg_error(name, "must be of length=4 and type list[int]")
    if any([el < 0 for el in arg]):
        config_arg_error(name, "elements must be positive integers")    
    return arg

E = TypeVar('E', bound=Enum)
def __cparse_enum(arg: Any, name: str, enum: Type[E]) -> E: # type: ignore the line
    if isinstance(arg, enum): return arg

    for enum_member in enum:
        if enum_member.value == arg:
            return enum_member
    config_arg_error(name, f"{arg} not a valid enum value")

def __cparse_IGNS(arg: Any, name: str) -> list[str]:
    if not isinstance(arg, list):
        config_arg_error(name, "not a list")
    if not 0 <= len(arg) <= 10:
        config_arg_error(name, "list must have a length of [0-10]")
    if not all([isinstance(el, str) for el in arg]):
        config_arg_error(name, "not a list of strings")
    if not all([1 < len(el) <= 18 for el in arg]): ## max length of an ign is 18?
        config_arg_error(name, "list contains an invalid IGN")
    return arg

def __cparse_file(arg: Any, name: str) -> str:
    if not isinstance(arg, str):
        config_arg_error(name, "not a string")
    if not exists(arg):
        config_arg_error(name, "file does not exist!")
    return arg


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
__deps_t: TypeAlias = dict[str, Any]
__config_deps: __deps_t = {
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
            "<|OPTION|>":   OPTIONAL,
        }
    },
}

__regiontool_deps: __deps_t = {
    "CAPTURE": {
        "FILE": [lambda cfg: cfg.capture.mode != CaptureMode.SCREENSHOT],
    }
}

__regiontool_defaults = {
    "CAPTURE": {
        "REGIONS": {}
    }
}


class Config:
    DEFAULTS_FILENAME = "defaults.json"
    DONT_SAVE = ["cfg_file_path", "name"]

    ## for linter
    scrim: bool
    spectator: bool
    capture: 'Config'
    mode: CaptureMode
    file: str
    regions: 'Config'
    timer: list[int]
    kf_line: list[int]
    team1_score: list[int]
    team2_score: list[int]
    team1_side: list[int]
    team2_side: list[int]
    igns: list[str]
    ign_mode: IGNMatrixMode
    max_rounds: int
    rounds_per_side: int
    screenshot_resize: int
    screenshot_period: float

    cfg_file_path: str

    def __init__(self, _inital: dict, *, name: Optional[str] = None) -> None:
        for key, value in _inital.items():
            if isinstance(value, dict):
                value = Config(value)
            self.__setattr__(Config.__fixkey(key), value)

        if name is not None:
            self.name = name
    
    def __setattr__(self, key: str, value: Any) -> None:
        super().__setattr__(key, value)

    def __getitem__(self, key: str) -> Any:
        return getattr(self, Config.__fixkey(key))

    def __setitem__(self, key: str, value: Any) -> None:
        setattr(self, Config.__fixkey(key), value)

    def __contains__(self, key: str) -> bool:
        return Config.__fixkey(key) in self.__dict__

    def get(self, key: str, _default: Any = None) -> Any:
        key = Config.__fixkey(key)
        if key in self:
            return self.__dict__[key]
        return _default

    @staticmethod
    def __fixkey(key: str) -> str:
        return key.lower().replace(" ", "_")

    def __repr__(self) -> str:
        return self._repr(0)

    def _repr(self, indent: int) -> str:
        items = []
        for key, value in self.__dict__.items():
            if isinstance(value, Config):
                items.append(f'{" " * indent}{key}:\n{value._repr(indent + 2)}')
            else:
                items.append(f'{" " * indent}{key}: {value}')
        return '\n'.join(items)
    
    def enumerate(self) -> list[str]:
        key_paths = []
        
        for key, value in self.__dict__.items():
            if isinstance(value, Config):
                key_paths += [f"{key}/{val_enum}" for val_enum in value.enumerate()]
            else:
                key_paths.append(key)

        return key_paths

    def to_dict(self) -> dict:
        func = lambda v: v.to_dict() if isinstance(v, Config) else Config._to_dict(v)
        return {k: func(v) for k,v in self.__dict__.items()}

    @staticmethod
    def _to_dict(value: Any) -> Any:
        if isinstance(value, Enum):
            return str(value)
        else:
            return value

    def save(self, file_path: str) -> None:
        for key in Config.DONT_SAVE:
            self.__dict__.pop(key)
        with open(file_path, "w") as f_out:
            json.dump(self.to_dict(), f_out, indent=4)


def create_config(config_filepath: str, args: argparse.Namespace) -> Config | list[Config]:
    ## argument checks
    if not (exists(config_filepath) or exists(join("configs", config_filepath))):
        exit(f"CONFIG ERROR: File '{config_filepath}' cannot be found!")

    if not exists(config_filepath):
        config_filepath = join("configs", config_filepath)

    ## Load config,defaults from json
    config = load_json(config_filepath)

    spec = __config_spec
    deps = __config_deps
    default_cfg = Config(load_json(Config.DEFAULTS_FILENAME), name="defaults")
    if args.region_tool:
        spec = __regiontool_spec
        deps = __regiontool_deps
        default_cfg = Config(__regiontool_defaults, name="defaults")

    return __validate_config(config, default_cfg, spec, deps, config_filepath)


## ----- HELPER FUNCTIONS -----
## Config Parser helper function
def __validate_config(config: list|dict, default_cfg: Config,
                      spec: __spec_t, deps: __deps_t,
                      config_filepath: str) -> Config | list[Config]:
    if isinstance(config, list):
        if len(config) == 1:
            config = config[0]
        elif len(config) > 1:
            config[0]["CFG_FILE_PATH"] = config_filepath
            cfg0 = __check_config_spec(Config(config[0], name="config0"), default_cfg, spec, deps)
            other_cfgs = [__infer_config(Config(cfg, name=f"config{i}"), cfg0)
                          for i, cfg in enumerate(config[1:], start=1)]
            return [cfg0] + other_cfgs
        else:
            raise TypeError("CONFIG ERROR: Config list is empty")

    if isinstance(config, dict):
        ## Pass config,spec,defaults to recursive validator function
        config["CFG_FILE_PATH"] = config_filepath
        return __check_config_spec(Config(config, name="config"), default_cfg, spec, deps)

    else:
        raise TypeError("CONFIG ERROR: Invalid config root type")


def __check_config_spec(config: Config, default_cfg: Config,
                        spec: __spec_t, deps: __deps_t,
                        key_path: str = "",
                        infer_list: Optional[list[tuple]] = None,
                        dep_list: Optional[list[tuple]] = None) -> Config:
    """Validates and parses the structure and values in the config given the spec"""
    first = False
    if infer_list is None or dep_list is None:
        infer_list = []
        dep_list = []
        first = True

    for skey, svalue in spec.items():
        ## eg. IGNS, CAPTURE/MODE, CAPTURE/REGIONS/TIMER
        nkey_path = f"{key_path}/{skey}".strip(r"/")

        if skey == "<|OPTION|>": continue
        if skey in config:
            if isinstance(config[skey], Config):
                assert isinstance(svalue, dict), f"CONFIG ERROR: {nkey_path} must be a dict not value"
                config[skey] = __check_config_spec(config[skey], default_cfg.get(skey, {}), svalue, deps.get(skey, {}),
                                                   nkey_path, infer_list, dep_list)
            else:
                assert not isinstance(svalue, dict), f"CONFIG ERROR: {nkey_path} must be a value not dict"
                assert len(svalue) == 2, f"SPEC ERROR: {nkey_path} is not valid (<OPTION>, __parse_func)"
                config[skey] = svalue[1](arg=config[skey], name=nkey_path)

        else:
            if isinstance(svalue, dict):
                assert "<|OPTION|>" in svalue, f"SPEC ERROR: The config spec has not set the <|OPTION|> for a sub-config"
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

    if not first:
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


def __infer_config(config: Config, src_cfg: Config) -> Config:
    return config


if __name__ == "__main__":
    print("Please run R6Analyser from run.py")
