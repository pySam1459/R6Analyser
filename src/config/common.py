import os

from pathlib import Path
from re import match

from pydantic import BaseModel, ConfigDict, model_validator, field_validator
from typing import Any, Optional, Self

from utils import Timestamp
from utils.constants import TIMESTAMP_PATTERN
from utils.enums import CaptureMode

from .utils import VALID_URLS


class CaptureCommon(BaseModel):
    mode:    CaptureMode

    file:    Optional[Path]      = None
    url:     Optional[str]       = None
    offset:  Optional[Timestamp] = None

    model_config = ConfigDict(extra="ignore")

    @field_validator("offset", mode="before")
    @classmethod
    def parse_offset(cls, v: Any) -> Optional[Timestamp]:
        if v is None:
            return None
        if isinstance(v, int) and v >= 0:
            return Timestamp.from_int(v)
        elif isinstance(v, str) and match(TIMESTAMP_PATTERN, v):
            return Timestamp.from_str(v)

        raise ValueError(f"Cannot parse capture offset: {v}")


    @model_validator(mode="after")
    def file_exists(self) -> Self:
        if self.mode == CaptureMode.VIDEOFILE:
            if self.file is None:
                raise ValueError(f"videofile capture mode requires `file` field to be specified")
            elif not self.file.exists():
                raise ValueError(f"Video file {self.file} does not exist")
            elif not os.access(self.file, os.R_OK):
                raise ValueError(f"Permission Error: Invalid permissions to read video file {self.file}")
        return self

    @model_validator(mode="after")
    def validate_url(self) -> Self:
        if self.mode not in [CaptureMode.YOUTUBE, CaptureMode.TWITCH]:
            return self

        if self.url is None:
            raise ValueError(f"{self.mode.value} capture mode required `url` field to be specified")

        if self.mode not in VALID_URLS:
            return self

        valid_urls = VALID_URLS[self.mode]
        for vurl in valid_urls:
            if self.url.startswith(vurl):
                return self
        else:
            valid_urls_str = ", ".join(valid_urls)
            raise ValueError(f"Invalid {self.mode.value} URL: {self.url}\nIt must start with {valid_urls_str}")
