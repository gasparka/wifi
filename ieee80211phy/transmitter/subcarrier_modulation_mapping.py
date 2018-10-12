"""
The OFDM subcarriers shall be modulated by using BPSK, QPSK, 16-QAM, or 64-QAM, depending on the
RATE requested. The encoded and interleaved binary serial input data shall be divided into groups of NBPSC
(1, 2, 4, or 6) bits and converted into complex numbers representing BPSK, QPSK, 16-QAM, or 64-QAM
constellation points. The conversion shall be performed according to Gray-coded constellation mappings.
"""

# For BPSK, LUT[B0] determines the I value, Q is always 0
BPSK_LUT = [-1, 1]

# For QPSK, LUT[B0] determines the I value and LUT[B1] determines the Q value
QPSK_LUT = [-1, 1]

# For 16-QAM, LUT[B0 B1] determines the I value and LUT[B2 B3] determines the Q value
QAM16_LUT = [-3, -1, 3, 1]

# For 64-QAM, LUT[B0 B1 B2] determines the I value and LUT[B3 B4 B5] determines the Q value
QAM64_LUT = [-7, -5, -1, -3, 7, 5, 1, 3]

"""
The normalization factor depends on the base modulation mode. Note that the modulation type can be different 
from the start to the end of the transmission, as the signal changes from SIGNAL to DATA. 
The purpose of the normalization factor is to achieve the same average power for all mappings.
"""
import numpy as np

BPSK_LUT_NORM = np.array(BPSK_LUT)
QPSK_LUT_NORM = np.array(QPSK_LUT) / np.sqrt(2)
QAM16_LUT_NORM = np.array(QAM16_LUT) / np.sqrt(10)
QAM64_LUT_NORM = np.array(QAM64_LUT) / np.sqrt(42)


def mapper(bits, bits_per_symbol=1):
    from textwrap import wrap
    bits = wrap(bits, bits_per_symbol)
    out = []
    for chunk in bits:
        if bits_per_symbol == 1:  # BPSK
            i_index = int(chunk[0], 2)
            symbol = BPSK_LUT_NORM[i_index] + 0.0j
        elif bits_per_symbol == 2:  # QPSK
            i_index = int(chunk[0], 2)
            q_index = int(chunk[1], 2)
            symbol = QPSK_LUT_NORM[i_index] + QPSK_LUT_NORM[q_index] * 1j
        elif bits_per_symbol == 4:  # QAM16
            i_index = int(chunk[0:2], 2)
            q_index = int(chunk[2:4], 2)
            symbol = QAM16_LUT_NORM[i_index] + QAM16_LUT_NORM[q_index] * 1j
        elif bits_per_symbol == 6:  # QAM64
            i_index = int(chunk[0:3], 2)
            q_index = int(chunk[3:6], 2)
            symbol = QAM64_LUT_NORM[i_index] + QAM64_LUT_NORM[q_index] * 1j

        out.append(symbol)
    return out


def test_i163():
    """ IEEE Std 802.11-2016 - Table I-19—Interleaved bits of first DATA symbol """

    # IEEE Std 802.11-2016 - Table I-19—Interleaved bits of first DATA symbol
    input = '011101111111000011101111110001000111001100000000101111110001000100010000100110100001110100010010011011100011100011110101011010010001101101101011100110000100001100000000000011011011001101101101'
    symbols = mapper(input, bits_per_symbol=4)

    # expected = [0j, 0j, 0j, 0j, 0j, 0j, (-0.316+0.316j), (-0.316+0.316j), (0.316+0.316j), (-0.949-0.949j), (0.316+0.949j), (1+0j), (0.316+0.316j), (0.316-0.949j), (-0.316-0.949j), (-0.316+0.316j), (-0.949+0.316j), (-0.949-0.949j), (-0.949-0.949j), (0.949+0.316j), (0.316+0.316j), (-0.949-0.316j), (-0.949-0.316j), (-0.949-0.316j), (-0.949-0.949j), (1+0j), (0.949-0.316j), (0.949+0.949j), (-0.949-0.316j), (0.316-0.316j), (-0.949-0.316j), (-0.949+0.949j), 0j, (-0.316+0.949j), (0.316+0.949j), (-0.949+0.316j), (0.949-0.949j), (0.316+0.316j), (-0.316-0.316j), (1+0j), (-0.316+0.949j), (0.949-0.316j), (-0.949-0.316j), (0.949+0.316j), (-0.316+0.949j), (0.949+0.316j), (0.949-0.316j), (0.949-0.949j), (-0.316-0.949j), (-0.949+0.316j), (-0.949-0.949j), (-0.949-0.949j), (-0.949-0.949j), (-1+0j), (0.316-0.316j), (0.949+0.316j), (-0.949+0.316j), (-0.316+0.949j), (0.316-0.316j), 0j, 0j, 0j, 0j, 0j]

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

    np.testing.assert_equal(expected, np.round(symbols, 3))


def test_i144():
    # Table I-9—SIGNAL field bits after interleaving
    input = '100101001101000000010100100000110010010010010100'
    symbols = mapper(input, bits_per_symbol=1)

    # Table I-10—Frequency domain representation of SIGNAL field
    expected = [(1 + 0j), (-1 + 0j), (-1 + 0j), (1 + 0j), (-1 + 0j), (1 + 0j),
                (-1 + 0j), (-1 + 0j), (1 + 0j), (1 + 0j), (-1 + 0j), (1 + 0j), (-1 + 0j), (-1 + 0j), (-1 + 0j),
                (-1 + 0j), (-1 + 0j), (-1 + 0j), (-1 + 0j), (1 + 0j), (-1 + 0j), (1 + 0j), (-1 + 0j),
                (-1 + 0j), (1 + 0j), (-1 + 0j), (-1 + 0j), (-1 + 0j), (-1 + 0j), (-1 + 0j), (1 + 0j),
                (1 + 0j), (-1 + 0j), (-1 + 0j), (1 + 0j), (-1 + 0j), (-1 + 0j), (1 + 0j), (-1 + 0j), (-1 + 0j),
                (1 + 0j), (-1 + 0j), (-1 + 0j), (1 + 0j), (-1 + 0j), (1 + 0j), (-1 + 0j), (-1 + 0j)]
    # expected = [0j, 0j, 0j, 0j, 0j, 0j, (1+0j), (-1+0j), (-1+0j), (1+0j), (-1+0j), (1+0j), (1+0j), (-1+0j), (-1+0j), (1+0j), (1+0j), (-1+0j), (1+0j), (-1+0j), (-1+0j), (-1+0j), (-1+0j), (-1+0j), (-1+0j), (1+0j), (-1+0j), (1+0j), (-1+0j), (1+0j), (-1+0j), (-1+0j), 0j, (1+0j), (-1+0j), (-1+0j), (-1+0j), (-1+0j), (-1+0j), (1+0j), (1+0j), (1+0j), (-1+0j), (-1+0j), (1+0j), (-1+0j), (-1+0j), (1+0j), (-1+0j), (-1+0j), (1+0j), (-1+0j), (-1+0j), (-1+0j), (1+0j), (-1+0j), (1+0j), (-1+0j), (-1+0j), 0j, 0j, 0j, 0j, 0j]
    np.testing.assert_equal(expected, np.round(symbols, 3))
