from typing import List
import numpy as np
from hypothesis import given
from hypothesis._strategies import lists, complex_numbers
from wifi.to_time_domain import OFDMFrame

SIZE = 16


def do(frames: List[OFDMFrame]) -> List[OFDMFrame]:
    return [frame[-SIZE:] + frame for frame in frames]


def undo(frames: List[OFDMFrame]) -> List[OFDMFrame]:
    return [frame[SIZE:] for frame in frames]


@given(lists(lists(complex_numbers(), min_size=64, max_size=64), min_size=1, max_size=32))
def test_hypothesis(data):
    un = undo(do(data))
    np.testing.assert_equal(data, un)
