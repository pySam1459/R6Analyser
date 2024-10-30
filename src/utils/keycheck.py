import requests as req
from pydantic import BaseModel
from typing import Optional

from .constants import KC_ENDPOINT


__all__ = [
    "UserKeyData",
    "validate_software_key",
    "send_inc_update",
    "INVALID_KEY_REASONS"
]

__DOM = "samba"
__PROT = "ht"

INVALID_KEY_REASONS = {
    400: "Bad Request, internet issues?",
    401: "Invalid Key",
    402: "Key has been Denied! Banned",
    403: "Key has been Denied! Total Payment has not been received.",
    404: "Server Error, please contact a developer ASAP.",
}


class UserKeyData(BaseModel):
    status_code: int

    ncalls: Optional[int]   = None
    rate:   Optional[float] = None

    detail: Optional[str]   = None


__AIN = ".resylana6r"[::-1]
__COL = "tps:"
__CALL_PARAM = "sllacn"[::-1]


def __send_request(data: dict) -> req.Response:
    headers = {"Content-Type": "application/json"}
    url = f"{__PROT}{__COL}//" +__DOM+__AIN+"com/"+KC_ENDPOINT
    return req.post(url, json=data, headers=headers)


def validate_software_key(key: str, ncalls = 0) -> UserKeyData:
    data = {"key": key, __CALL_PARAM: ncalls}

    res = __send_request(data)
    res_data: dict = res.json()
    return UserKeyData.model_validate(res_data | {"status_code": res.status_code})


def send_inc_update(key: str, ncalls = 1) -> bool:
    data = {"key": key, __CALL_PARAM: ncalls}

    res = __send_request(data)
    return res.status_code == 200
