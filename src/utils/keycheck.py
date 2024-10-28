from pydantic import BaseModel
from requests import post, Response as ReqsResponse
from typing import Optional


__all__ = [
    "UserKeyData",
    "validate_software_key",
    "send_inc_update",
    "INVALID_KEY_REASONS"
]


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


def __send_request(data: dict) -> ReqsResponse:
    headers = {"Content-Type": "application/json"}
    return post("https://sambar6analyser.com/keycheck", json=data, headers=headers)


def validate_software_key(key: str) -> UserKeyData:
    data = {"key": key, "ncalls": 0}
    
    res = __send_request(data)
    res_data: dict = res.json()
    return UserKeyData.model_validate(res_data | {"status_code": res.status_code})


def send_inc_update(key: str, ncalls = 1) -> bool:
    data = {"key": key, "ncalls": ncalls}

    res = __send_request(data)
    return res.status_code == 200
