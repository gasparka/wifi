from typing import List
import numpy as np
from hypothesis import given
from hypothesis._strategies import lists, complex_numbers
from wifi.subcarrier_mapping import Carriers

PILOT_POLARITY = [1, 1, 1, 1, -1, -1, -1, 1, -1, -1, -1, -1, 1, 1, -1, 1, -1, -1, 1, 1, -1, 1, 1, -1, 1, 1, 1, 1, 1, 1,
                  -1, 1, 1, 1, -1, 1, 1, -1, -1, 1, 1, 1, -1, 1, -1, -1, -1, 1, -1, 1, -1, -1, 1, -1, -1, 1, 1, 1, 1, 1,
                  -1, -1, 1, 1, -1, -1, 1, -1, 1, -1, 1, 1, -1, -1, -1, 1, 1, -1, -1, -1, -1, 1, -1, -1, 1, -1, 1, 1, 1,
                  1, -1, 1, -1, 1, -1, 1, -1, -1, -1, -1, -1, 1, -1, 1, 1, -1, 1, -1, 1, 1, 1, -1, -1, 1, -1, -1, -1, 1,
                  1, 1, -1, -1, -1, -1, -1, -1, -1]


def do_one(carriers: Carriers, index_in_package: int) -> Carriers:
    pilots = np.array([1, 1, 1, -1], dtype=complex) * PILOT_POLARITY[index_in_package % 127]
    carriers[-21] = pilots[0]
    carriers[-7] = pilots[1]
    carriers[7] = pilots[2]
    carriers[21] = pilots[3]

    return carriers


def undo_one(carriers: Carriers, index_in_package: int) -> Carriers:

    pilots = np.empty(4, dtype=complex)
    pilots[0] = carriers[-21]
    pilots[1] = carriers[-7]
    pilots[2] = carriers[7]
    pilots[3] = carriers[21]

    # remove latent frequency offset by using pilot symbols
    pilots *= PILOT_POLARITY[index_in_package % 127]
    mean_phase_offset = np.angle(np.mean(pilots))
    carriers = np.array(carriers) * np.exp(-1j * mean_phase_offset)

    return carriers.tolist()


def do(carriers: List[Carriers]) -> List[Carriers]:
    return [do_one(carrier, index) for index, carrier in enumerate(carriers)]


def undo(carriers: List[Carriers]) -> List[Carriers]:
    return [undo_one(carrier, index) for index, carrier in enumerate(carriers)]


@given(lists(lists(complex_numbers(allow_nan=False, allow_infinity=False), min_size=64, max_size=64), min_size=1, max_size=32))
def test_hypothesis(data):
    un = undo(do(data))
    np.testing.assert_allclose(data, un, rtol=1e-16, atol=1e-16)
