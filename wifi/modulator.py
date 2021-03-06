"""
The OFDM subcarriers shall be modulated by using BPSK, QPSK, 16-QAM, or 64-QAM, depending on the
RATE requested. The encoded and interleaved binary serial input data shall be divided into groups of NBPSC
(1, 2, 4, or 6) bits and converted into complex numbers representing BPSK, QPSK, 16-QAM, or 64-QAM
constellation points. The conversion shall be performed according to Gray-coded constellation mappings.

The normalization factor depends on the base modulation mode. Note that the modulation type can be different
from the start to the end of the transmission, as the signal changes from SIGNAL to DATA.
The purpose of the normalization factor is to achieve the same average power for all mappings.
"""
from typing import List, NewType
import numpy as np
from hypothesis import given
from hypothesis._strategies import binary, sampled_from
from wifi import bitstr
from wifi.bitstr import bits

BPSK_LUT = np.array([-1 + 0j, 1 + 0j])

QPSK_LUT = np.array([-1 - 1j, -1 + 1j,
                     1 - 1j, 1 + 1j]) / np.sqrt(2)

QAM16_LUT = np.array([-3 - 3j, -3 - 1j, -3 + 3j, -3 + 1j,
                      -1 - 3j, -1 - 1j, -1 + 3j, -1 + 1j,
                      3 - 3j, 3 - 1j, 3 + 3j, 3 + 1j,
                      1 - 3j, 1 - 1j, 1 + 3j, 1 + 1j]) / np.sqrt(10)

QAM64_LUT = np.array([-7 - 7j, -7 - 5j, -7 - 1j, -7 - 3j, -7 + 7j, -7 + 5j, -7 + 1j, -7 + 3j,
                      -5 - 7j, -5 - 5j, -5 - 1j, -5 - 3j, -5 + 7j, -5 + 5j, -5 + 1j, -5 + 3j,
                      -1 - 7j, -1 - 5j, -1 - 1j, -1 - 3j, -1 + 7j, -1 + 5j, -1 + 1j, -1 + 3j,
                      -3 - 7j, -3 - 5j, -3 - 1j, -3 - 3j, -3 + 7j, -3 + 5j, -3 + 1j, -3 + 3j,
                      7 - 7j, 7 - 5j, 7 - 1j, 7 - 3j, 7 + 7j, 7 + 5j, 7 + 1j, 7 + 3j,
                      5 - 7j, 5 - 5j, 5 - 1j, 5 - 3j, 5 + 7j, 5 + 5j, 5 + 1j, 5 + 3j,
                      1 - 7j, 1 - 5j, 1 - 1j, 1 - 3j, 1 + 7j, 1 + 5j, 1 + 1j, 1 + 3j,
                      3 - 7j, 3 - 5j, 3 - 1j, 3 - 3j, 3 + 7j, 3 + 5j, 3 + 1j, 3 + 3j]) / np.sqrt(42)

# maps 'bits_per_symbol' to correct modulation table
LUT = {1: BPSK_LUT,
       2: QPSK_LUT,
       4: QAM16_LUT,
       6: QAM64_LUT}


Symbol = NewType('Symbol', complex)
Symbol.__doc__ = """ Frequency domain value, used to modulate individual OFDM carrier """


def do(data: bits, bits_per_symbol: int) -> List[Symbol]:
    indexes = [bitstr.to_int(group) for group in bitstr.split(data, bits_per_symbol)]
    symbols = [LUT[bits_per_symbol][index] for index in indexes]
    return symbols


def undo(symbols: List[Symbol], bits_per_symbol: int) -> bits:
    errors = [abs(LUT[bits_per_symbol] - symbol) for symbol in symbols]
    best_indexes = np.argmin(errors, axis=1)
    res = [bitstr.from_int(index, bits_per_symbol) for index in best_indexes]
    return bitstr.merge(res)


def symbols_error(symbols: np.ndarray, bits_per_symbol: int) -> np.ndarray:
    # TODO: wtf is this?
    errors = [abs(LUT[bits_per_symbol] - symbol) for symbol in symbols]
    ret = [np.min(err) for err in errors]
    return np.array(ret)


def test_i163():
    """ IEEE Std 802.11-2016 - Table I-19—Interleaved bits of first DATA symbol """

    # IEEE Std 802.11-2016 - Table I-19—Interleaved bits of first DATA symbol
    input = bits('01110111111100001110111111000100011100110000000010111111000100010001000010011010000111010001001'
                 '0011011100011100011110101011010010001101101101011100110000100001100000000000011011011001101101101')

    # Table I-20—Frequency domain of first DATA symbol
    # I have removed padding, tail and pilot symbols - these will be added in next block
    expected = [(-0.316 + 0.316j), (-0.316 + 0.316j), (0.316 + 0.316j), (-0.949 - 0.949j), (0.316 + 0.949j),
                (0.316 + 0.316j), (0.316 - 0.949j), (-0.316 - 0.949j), (-0.316 + 0.316j), (-0.949 + 0.316j),
                (-0.949 - 0.949j), (-0.949 - 0.949j), (0.949 + 0.316j), (0.316 + 0.316j), (-0.949 - 0.316j),
                (-0.949 - 0.316j), (-0.949 - 0.316j), (-0.949 - 0.949j), (0.949 - 0.316j), (0.949 + 0.949j),
                (-0.949 - 0.316j), (0.316 - 0.316j), (-0.949 - 0.316j), (-0.949 + 0.949j), (-0.316 + 0.949j),
                (0.316 + 0.949j), (-0.949 + 0.316j), (0.949 - 0.949j), (0.316 + 0.316j), (-0.316 - 0.316j),
                (-0.316 + 0.949j), (0.949 - 0.316j), (-0.949 - 0.316j), (0.949 + 0.316j), (-0.316 + 0.949j),
                (0.949 + 0.316j), (0.949 - 0.316j), (0.949 - 0.949j), (-0.316 - 0.949j), (-0.949 + 0.316j),
                (-0.949 - 0.949j), (-0.949 - 0.949j), (-0.949 - 0.949j), (0.316 - 0.316j), (0.949 + 0.316j),
                (-0.949 + 0.316j), (-0.316 + 0.949j), (0.316 - 0.316j)]

    symbols = do(input, bits_per_symbol=4)
    np.testing.assert_equal(expected, np.round(symbols, 3))

    # test reverse
    rev = undo(symbols, bits_per_symbol=4)
    assert rev == input


def test_i144():
    # Table I-9—SIGNAL field bits after interleaving
    input = bits('100101001101000000010100100000110010010010010100')

    # Table I-10—Frequency domain representation of SIGNAL field
    expected = [(1 + 0j), (-1 + 0j), (-1 + 0j), (1 + 0j), (-1 + 0j), (1 + 0j),
                (-1 + 0j), (-1 + 0j), (1 + 0j), (1 + 0j), (-1 + 0j), (1 + 0j), (-1 + 0j), (-1 + 0j), (-1 + 0j),
                (-1 + 0j), (-1 + 0j), (-1 + 0j), (-1 + 0j), (1 + 0j), (-1 + 0j), (1 + 0j), (-1 + 0j),
                (-1 + 0j), (1 + 0j), (-1 + 0j), (-1 + 0j), (-1 + 0j), (-1 + 0j), (-1 + 0j), (1 + 0j),
                (1 + 0j), (-1 + 0j), (-1 + 0j), (1 + 0j), (-1 + 0j), (-1 + 0j), (1 + 0j), (-1 + 0j), (-1 + 0j),
                (1 + 0j), (-1 + 0j), (-1 + 0j), (1 + 0j), (-1 + 0j), (1 + 0j), (-1 + 0j), (-1 + 0j)]

    symbols = do(input, bits_per_symbol=1)
    np.testing.assert_equal(expected, np.round(symbols, 3))

    # test reverse
    rev = undo(symbols, bits_per_symbol=1)
    assert rev == input


@given(binary(), sampled_from([1, 2, 4, 6]))
def test_hypothesis(data, bits_per_symbol):
    data = bitstr.from_bytes(data)
    assert undo(do(data, bits_per_symbol), bits_per_symbol) == data

