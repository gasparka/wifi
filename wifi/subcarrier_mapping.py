"""
Divide the complex number string into groups of 48 complex numbers. Each such group is
associated with one OFDM symbol. In each group, the complex numbers are numbered 0 to 47 and
mapped hereafter into OFDM subcarriers numbered -26 to -22, -20 to -8, -6 to -1, 1 to 6, 8 to 20,
and 22 to 26. The subcarriers -21, -7, 7, and 21 are skipped and, subsequently, used for inserting
pilot subcarriers. The 0 subcarrier, associated with center frequency, is omitted and filled with the
value 0. Refer to 17.3.5.10 for details.

Clarification: IFFT of 64 points are used. 802.11 uses 48 subcarriers + 4 pilot tones, thus leaving
12 empty tones. This is solved by setting -32..-27, 0 and 27..31 subcarriers to 0+0j.
"""

from typing import List
import numpy as np
from hypothesis import assume, given
from hypothesis._strategies import lists
from hypothesis.extra.numpy import arrays
from more_itertools import chunked, flatten
from wifi import modulator
from wifi.modulator import Symbol

Carriers = List[modulator.Symbol]
Carriers.__doc__ = """ Contains 64 Symbols, one to modulate each OFDM carrier. """


def do_one(symbols: List[Symbol]) -> Carriers:
    assume(len(symbols) == 48)

    carriers = np.empty(64, dtype=complex)
    carriers[-32] = 0j
    carriers[-31] = 0j
    carriers[-30] = 0j
    carriers[-29] = 0j
    carriers[-28] = 0j
    carriers[-27] = 0j
    carriers[-26] = symbols[0]
    carriers[-25] = symbols[1]
    carriers[-24] = symbols[2]
    carriers[-23] = symbols[3]
    carriers[-22] = symbols[4]
    # carriers[-21] = pilots[0]
    carriers[-20] = symbols[5]
    carriers[-19] = symbols[6]
    carriers[-18] = symbols[7]
    carriers[-17] = symbols[8]
    carriers[-16] = symbols[9]
    carriers[-15] = symbols[10]
    carriers[-14] = symbols[11]
    carriers[-13] = symbols[12]
    carriers[-12] = symbols[13]
    carriers[-11] = symbols[14]
    carriers[-10] = symbols[15]
    carriers[-9] = symbols[16]
    carriers[-8] = symbols[17]
    # carriers[-7] = pilots[1]
    carriers[-6] = symbols[18]
    carriers[-5] = symbols[19]
    carriers[-4] = symbols[20]
    carriers[-3] = symbols[21]
    carriers[-2] = symbols[22]
    carriers[-1] = symbols[23]
    carriers[0] = 0j
    carriers[1] = symbols[24]
    carriers[2] = symbols[25]
    carriers[3] = symbols[26]
    carriers[4] = symbols[27]
    carriers[5] = symbols[28]
    carriers[6] = symbols[29]
    # carriers[7] = pilots[2]
    carriers[8] = symbols[30]
    carriers[9] = symbols[31]
    carriers[10] = symbols[32]
    carriers[11] = symbols[33]
    carriers[12] = symbols[34]
    carriers[13] = symbols[35]
    carriers[14] = symbols[36]
    carriers[15] = symbols[37]
    carriers[16] = symbols[38]
    carriers[17] = symbols[39]
    carriers[18] = symbols[40]
    carriers[19] = symbols[41]
    carriers[20] = symbols[42]
    # carriers[21] = pilots[3]
    carriers[22] = symbols[43]
    carriers[23] = symbols[44]
    carriers[24] = symbols[45]
    carriers[25] = symbols[46]
    carriers[26] = symbols[47]
    carriers[27] = 0j
    carriers[28] = 0j
    carriers[29] = 0j
    carriers[30] = 0j
    carriers[31] = 0j

    return list(carriers)


def undo_one(carriers: Carriers) -> List[Symbol]:
    symbols = np.empty(48, dtype=complex)
    symbols[0] = carriers[-26]
    symbols[1] = carriers[-25]
    symbols[2] = carriers[-24]
    symbols[3] = carriers[-23]
    symbols[4] = carriers[-22]
    # pilots[0] = carriers[-21]
    symbols[5] = carriers[-20]
    symbols[6] = carriers[-19]
    symbols[7] = carriers[-18]
    symbols[8] = carriers[-17]
    symbols[9] = carriers[-16]
    symbols[10] = carriers[-15]
    symbols[11] = carriers[-14]
    symbols[12] = carriers[-13]
    symbols[13] = carriers[-12]
    symbols[14] = carriers[-11]
    symbols[15] = carriers[-10]
    symbols[16] = carriers[-9]
    symbols[17] = carriers[-8]
    # pilots[1] = carriers[-7]
    symbols[18] = carriers[-6]
    symbols[19] = carriers[-5]
    symbols[20] = carriers[-4]
    symbols[21] = carriers[-3]
    symbols[22] = carriers[-2]
    symbols[23] = carriers[-1]
    symbols[24] = carriers[1]
    symbols[25] = carriers[2]
    symbols[26] = carriers[3]
    symbols[27] = carriers[4]
    symbols[28] = carriers[5]
    symbols[29] = carriers[6]
    # pilots[2] = carriers[7]
    symbols[30] = carriers[8]
    symbols[31] = carriers[9]
    symbols[32] = carriers[10]
    symbols[33] = carriers[11]
    symbols[34] = carriers[12]
    symbols[35] = carriers[13]
    symbols[36] = carriers[14]
    symbols[37] = carriers[15]
    symbols[38] = carriers[16]
    symbols[39] = carriers[17]
    symbols[40] = carriers[18]
    symbols[41] = carriers[19]
    symbols[42] = carriers[20]
    # pilots[3] = carriers[21]
    symbols[43] = carriers[22]
    symbols[44] = carriers[23]
    symbols[45] = carriers[24]
    symbols[46] = carriers[25]
    symbols[47] = carriers[26]

    return list(symbols)


def do(symbols: List[Symbol]) -> List[Carriers]:
    return [do_one(chunk) for chunk in chunked(symbols, 48)]


def undo(carriers: List[Carriers]) -> List[Symbol]:
    return list(flatten([undo_one(carrier) for carrier in carriers]))


@given(lists(arrays(complex, 48), min_size=1))
def test_hypothesis(data):
    data = np.hstack(data).tolist()
    un = undo(do(data))
    np.testing.assert_equal(data, un)
