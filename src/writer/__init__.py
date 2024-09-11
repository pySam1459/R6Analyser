from pathlib import Path

from config import Config
from utils.enums import SaveFileType

from .base import Writer
from .xlsx import XlsxWriter
from .json import JsonWriter


def get_save_path(config: Config) -> Path:
    if config.save.path is not None:
        return config.save.path
    
    return config.save.save_dir / config.name / config.save.file_type


def create_writer(config: Config) -> Writer:
    save_path = get_save_path(config)

    match config.save.file_type:
        case SaveFileType.XLSX:
            return XlsxWriter(save_path, config)
        case SaveFileType.JSON:
            return JsonWriter(save_path)
