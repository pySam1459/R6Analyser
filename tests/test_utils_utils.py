import pytest

from utils import Timestamp


testdata_timestamp = [
    (180, (3,0)),(120, (2,0)),(60,  (1,0)),(0,   (0,0)),
    (179, (2,59)),(119, (1,59)),(59, (0,59)),
    (30, (0, 30)),(90, (1, 30)),(150, (2,30)),
    (-45, (0, -45)),(-10, (0,-10)),(-60, (-1, 0)),(-90, (-1, -30)),
    (1200, (20,0)),(600, (10,0)),(430, (7,10))
]
ids_timestamp = [f"{inp} -> {m}m {s}s" for inp, (m,s) in testdata_timestamp]

@pytest.mark.parametrize("input_num,min_sec", testdata_timestamp, ids=ids_timestamp)
def test_timestamp(input_num, min_sec) -> None:
    out = Timestamp.from_int(input_num)

    assert out.minutes == min_sec[0]
    assert out.seconds == min_sec[1]

    assert out.to_int() == input_num

    if min_sec[1] < 0:
        assert str(out) == f"{min_sec[0]}:{min_sec[1]:03}"
    else:
        assert str(out) == f"{min_sec[0]}:{min_sec[1]:02}"


testdata_scoreline = []
ids_scoreline = []