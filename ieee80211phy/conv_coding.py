import logging
from multiprocessing.pool import Pool
from textwrap import wrap
import numpy as np
from numba import jit

from ieee80211phy.util import int_to_binstr, xor_reduce_poly

logger = logging.getLogger(__name__)

# config
K = 7
STATES = 2 ** (K - 1)
G0 = int('133', 8)
G1 = int('171', 8)

# input bit is considered as an additional state (MSb) in this LUT, thus it has (states * 2) values
OUTPUT_LUT = [(xor_reduce_poly(x, G0) << 1) | xor_reduce_poly(x, G1) for x in range(2 ** K)]


def _puncture(data, rate, undo=False):
    """
    Puncturing is a procedure for omitting some of the encoded bits in the transmitter
    (thus reducing the number of transmitted bits and increasing the coding rate)
    and inserting a dummy “zero” metric into the convolutional decoder on the
    receive side in place of the omitted bits. The puncturing patterns are illustrated in Figure 17-9.
    """

    if undo:
        # un-puncturing process i.e. add 'X' characters that are ignored by the error calculation
        if rate == '3/4':
            data = [d[:3] + 'XX' + d[3] for d in wrap(data, 4)]
        elif rate == '2/3':
            data = [d + 'X' for d in wrap(data, 3)]
    else:
        if rate == '2/3':
            # throw out ech 3. bit in a block of 4
            data = [bit for i, bit in enumerate(data) if (i % 4) != 3]
        elif rate == '3/4':
            # throw out each 3. and 4. bit in a block of 6 bits
            data = [bit for i, bit in enumerate(data) if (i % 6) != 3 and (i % 6) != 4]
    return ''.join(data)


def conv_encode(data, coding_rate='1/2'):
    """ See Figure 17-8—Convolutional encoder """
    output = ''
    shr = '0' * (K - 1)
    for bit in data:
        i = bit + shr
        output += int_to_binstr(OUTPUT_LUT[int(i, 2)], bits=2)
        shr = i[:-1]  # advance the shift register

    return _puncture(output, coding_rate)


# input_bit, parent1_id, parent1_output, parent2_id, parent2_output

LUTT = [
      ('0', 0, 0, 1, 3)
    , ('0', 2, 2, 3, 1)
    , ('0', 4, 0, 5, 3)
    , ('0', 6, 2, 7, 1)
    , ('0', 8, 3, 9, 0)
    , ('0', 10, 1, 11, 2)
    , ('0', 12, 3, 13, 0)
    , ('0', 14, 1, 15, 2)
    , ('0', 16, 3, 17, 0)
    , ('0', 18, 1, 19, 2)
    , ('0', 20, 3, 21, 0)
    , ('0', 22, 1, 23, 2)
    , ('0', 24, 0, 25, 3)
    , ('0', 26, 2, 27, 1)
    , ('0', 28, 0, 29, 3)
    , ('0', 30, 2, 31, 1)
    , ('0', 32, 1, 33, 2)
    , ('0', 34, 3, 35, 0)
    , ('0', 36, 1, 37, 2)
    , ('0', 38, 3, 39, 0)
    , ('0', 40, 2, 41, 1)
    , ('0', 42, 0, 43, 3)
    , ('0', 44, 2, 45, 1)
    , ('0', 46, 0, 47, 3)
    , ('0', 48, 2, 49, 1)
    , ('0', 50, 0, 51, 3)
    , ('0', 52, 2, 53, 1)
    , ('0', 54, 0, 55, 3)
    , ('0', 56, 1, 57, 2)
    , ('0', 58, 3, 59, 0)
    , ('0', 60, 1, 61, 2)
    , ('0', 62, 3, 63, 0)
    , ('1', 0, 3, 1, 0)
    , ('1', 2, 1, 3, 2)
    , ('1', 4, 3, 5, 0)
    , ('1', 6, 1, 7, 2)
    , ('1', 8, 0, 9, 3)
    , ('1', 10, 2, 11, 1)
    , ('1', 12, 0, 13, 3)
    , ('1', 14, 2, 15, 1)
    , ('1', 16, 0, 17, 3)
    , ('1', 18, 2, 19, 1)
    , ('1', 20, 0, 21, 3)
    , ('1', 22, 2, 23, 1)
    , ('1', 24, 3, 25, 0)
    , ('1', 26, 1, 27, 2)
    , ('1', 28, 3, 29, 0)
    , ('1', 30, 1, 31, 2)
    , ('1', 32, 2, 33, 1)
    , ('1', 34, 0, 35, 3)
    , ('1', 36, 2, 37, 1)
    , ('1', 38, 0, 39, 3)
    , ('1', 40, 1, 41, 2)
    , ('1', 42, 3, 43, 0)
    , ('1', 44, 1, 45, 2)
    , ('1', 46, 3, 47, 0)
    , ('1', 48, 1, 49, 2)
    , ('1', 50, 3, 51, 0)
    , ('1', 52, 1, 53, 2)
    , ('1', 54, 3, 55, 0)
    , ('1', 56, 2, 57, 1)
    , ('1', 58, 0, 59, 3)
    , ('1', 60, 2, 61, 1)
    , ('1', 62, 0, 63, 3)
]

ERR_LUT = [[0, 1, 1, 2, 0, 1, 0, 1],
           [1, 0, 2, 1, 0, 1, 1, 0],
           [1, 2, 0, 1, 1, 0, 0, 1],
           [2, 1, 1, 0, 1, 0, 1, 0]]




# @profile
def conv_decode(rx, coding_rate='1/2'):
    """ See 'Bits, Signals, and Packets: An Introduction to Digital Communications and Networks' ->
        'Viterbi Decoding of Convolutional Codes (PDF - 1.4MB)'
    """

    def butterfly(state, expected, scores):
        input_bit, parent1, parent1_out, parent2, parent2_out = LUTT[state]

        parent1_score = scores[parent1][0] + ERR_LUT[parent1_out][expected]
        parent2_score = scores[parent2][0] + ERR_LUT[parent2_out][expected]

        if parent1_score < parent2_score:
            return parent1_score, scores[parent1][1] + input_bit
        else:
            return parent2_score, scores[parent2][1] + input_bit

    rx = _puncture(rx, coding_rate, undo=True)
    symbols = ['00', '01', '10', '11', '0X', '1X', 'X0', 'X1']
    rx = [symbols.index(sym) for sym in wrap(rx, 2)]

    scores = [(0, '')] + ([(1000, '')] * (STATES - 1))  # (state score, decoded bits)
    for expect in rx:
        scores = [butterfly(i, expect, scores) for i in range(len(scores))]

    min_score_index = int(np.argmin([x[0] for x in scores]))
    bits = scores[min_score_index][1]
    logger.info(f'Decoded {len(bits)} bits, score={scores[min_score_index][0]}, rate={coding_rate}')
    return bits


def test_signal():
    """
    Uses 1/2 coding rate for signal field.
    """

    # IEEE Std 802.11-2016: Table I-7—Bit assignment for SIGNAL field
    input = '101100010011000000000000'

    # IEEE Std 802.11-2016: Table I-8—SIGNAL field bits after encoding
    expected = '110100011010000100000010001111100111000000000000'
    output = conv_encode(input, coding_rate='1/2')
    assert output == expected

    # test decoding
    decoded = conv_decode(output)
    assert decoded == input

    # test decoding with bit errors
    # 4 errors
    output = '010100011010100101000010001111100111000000100000'
    decoded = conv_decode(output)
    assert decoded == input


def test_i161():
    """
    Uses 3/4 coding rate for data field.
    """

    # Table I-15—The DATA bits after scrambling (and resetting tail bits)
    input = '0110110000011001100010011000111101101000001000011111010010100101011000010100111111010111101011100' \
            '0100100000011001111001100111010111001001011110001010011100110001100000000011110001101011011001111' \
            '100011111110000010010101100000110101100010010100110101001100111111111011110000010000010010101110' \
            '001111010100110001110010000011010000011011111000111001001010000110011001000100011001101100110111' \
            '110110101000111101100000001101110101001000000100111011001011111101111111000011010110001111011111' \
            '000110010100101110101101110000100011111001111001101010100100001000000111111101011111001010100111' \
            '01000101010101000100100000010001110100110110011110100100111011110011011001001110001101011110110' \
            '111110001110000000000100010000010011001101000010111110110001010001001110001011100111001000101011' \
            '01000001110110010010101000101101001000100010000000000001101110001111111000011101111001011001001'

    # Table I-16—The BCC encoded DATA bits
    expect = '0010101100001000101000011111000010011101101101011001101000011101010010101111101111101000110000101' \
             '0001111110000001100100001110011110000000100001111100000000110011110000011010011111010111011001010' \
             '10111110011000111111010101100100001111100010110110100101100110000011001010101011011001000100000101' \
             '01101000101110100110010000000110010010110011001000011001111010001110100100011100000100000101101101' \
             '11101101111100010111011000100000000010111110100010110111010110111100101011100101110110000111011001' \
             '11011101000011010001001001110110001001110000001001001100100100101011110000010010010010110110101011' \
             '1101110111000000100011001001111000111000000001101101001101011011000011011010100110000001001101010' \
             '0101111001011010100011001110010110000100000001110101110001001001101000001010001000000011011001001' \
             '11100010110001010001010010110111110001101001000100010001010110000110111111011011111001011110111110' \
             '01101111100010100011101111110010100101010100000011111111010010011010010001001110111001010111101100' \
             '01011011001000111001100101011111001010000011111011010100111010011111011110111000000100110111010110' \
             '001110111100101010000000011011011011001110100100000111010111011011000010111111'

    output = conv_encode(input, coding_rate='3/4')
    assert output == expect

    # test decoding
    decoded = conv_decode(expect, coding_rate='3/4')
    assert decoded == input
