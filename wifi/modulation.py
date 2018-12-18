"""
The OFDM subcarriers shall be modulated by using BPSK, QPSK, 16-QAM, or 64-QAM, depending on the
RATE requested. The encoded and interleaved binary serial input data shall be divided into groups of NBPSC
(1, 2, 4, or 6) bits and converted into complex numbers representing BPSK, QPSK, 16-QAM, or 64-QAM
constellation points. The conversion shall be performed according to Gray-coded constellation mappings.

The normalization factor depends on the base modulation mode. Note that the modulation type can be different
from the start to the end of the transmission, as the signal changes from SIGNAL to DATA.
The purpose of the normalization factor is to achieve the same average power for all mappings.
"""
import numpy as np
from textwrap import wrap
from wifi.util import int_to_binstr

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


def bits_to_symbols(bits: str, bits_per_symbol: int) -> np.ndarray:
    indexes = [int(bit_group, 2) for bit_group in wrap(bits, bits_per_symbol)]
    symbols = [LUT[bits_per_symbol][index] for index in indexes]
    return np.array(symbols)


def symbols_to_bits(symbols: np.ndarray, bits_per_symbol: int) -> str:
    errors = [abs(LUT[bits_per_symbol] - symbol) for symbol in symbols]
    indexes = np.argmin(errors, axis=1)
    bits = ''.join(int_to_binstr(index, bits=bits_per_symbol) for index in indexes)
    return bits


def symbols_error(symbols: np.ndarray, bits_per_symbol: int) -> np.ndarray:
    errors = [abs(LUT[bits_per_symbol] - symbol) for symbol in symbols]
    ret = [np.min(err) for err in errors]
    return np.array(ret)


def test_i163():
    """ IEEE Std 802.11-2016 - Table I-19—Interleaved bits of first DATA symbol """

    # IEEE Std 802.11-2016 - Table I-19—Interleaved bits of first DATA symbol
    input = '011101111111000011101111110001000111001100000000101111110001000100010000100110100001110100010010011011100011100011110101011010010001101101101011100110000100001100000000000011011011001101101101'

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

    symbols = bits_to_symbols(input, bits_per_symbol=4)
    np.testing.assert_equal(expected, np.round(symbols, 3))

    # test reverse
    rev = symbols_to_bits(symbols, bits_per_symbol=4)
    assert rev == input


def test_i144():
    # Table I-9—SIGNAL field bits after interleaving
    input = '100101001101000000010100100000110010010010010100'

    # Table I-10—Frequency domain representation of SIGNAL field
    expected = [(1 + 0j), (-1 + 0j), (-1 + 0j), (1 + 0j), (-1 + 0j), (1 + 0j),
                (-1 + 0j), (-1 + 0j), (1 + 0j), (1 + 0j), (-1 + 0j), (1 + 0j), (-1 + 0j), (-1 + 0j), (-1 + 0j),
                (-1 + 0j), (-1 + 0j), (-1 + 0j), (-1 + 0j), (1 + 0j), (-1 + 0j), (1 + 0j), (-1 + 0j),
                (-1 + 0j), (1 + 0j), (-1 + 0j), (-1 + 0j), (-1 + 0j), (-1 + 0j), (-1 + 0j), (1 + 0j),
                (1 + 0j), (-1 + 0j), (-1 + 0j), (1 + 0j), (-1 + 0j), (-1 + 0j), (1 + 0j), (-1 + 0j), (-1 + 0j),
                (1 + 0j), (-1 + 0j), (-1 + 0j), (1 + 0j), (-1 + 0j), (1 + 0j), (-1 + 0j), (-1 + 0j)]

    symbols = bits_to_symbols(input, bits_per_symbol=1)
    np.testing.assert_equal(expected, np.round(symbols, 3))

    # test reverse
    rev = symbols_to_bits(symbols, bits_per_symbol=1)
    assert rev == input
