from pathlib import Path
from re import compile


IGN_REGEX = compile(r"^[0-9a-zA-Z\-_\.]{2,18}$")
SOFTWARE_KEY_PATTERN = r"^[a-f0-9]{64}$"

DEBUG_PATH            = Path("debug.json")
DEFAULTS_PATH         = Path("defaults.json")
SETTINGS_PATH         = Path("settings.json")
DOTENV_PATH           = Path(".env")
ASSETS_PATH           = Path("assets")
DEFAULT_SAVE_DIR      = Path("saves")
DEFAULT_TESSDATA_PATH = Path("./tessdata")

DEFAULT_LANGUAGE = "en"

## Console text colour codes
RED = '\033[91m'
YLW = '\u001b[33m'
WHITE = '\033[97m'

## Red Thresholds (RT)
BOMB_COUNTDOWN_RT = 0.70
TIMER_LAST_SECONDS_RT = 0.65

DIGITS = "0123456789"
TIMER_CHATLIST = DIGITS + ":."
IGN_CHARLIST = "abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789.-_"

TRUTHY_STRINGS = ["true", "t", "yes", "y"]
FALSEY_STRINGS = ["false", "f", "no", "n"]

## IGNMAT
IM_LEVEN_THRESHOLD = 0.65 ## threshold for Levenshtein distance to determine equality
IM_TEAM_DET_THRESHOLD = 3


LANGUAGES = [
    "en"
]