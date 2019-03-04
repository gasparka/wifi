from loguru import logger
import numpy as np
from hypothesis import given, assume, settings
from hypothesis._strategies import binary
from numba import njit
from wifi import puncturer, bits
from wifi.util import xor_reduce_poly, is_divisible

# config
K = 7
STATES = 2 ** (K - 1)
G0 = int('133', 8)
G1 = int('171', 8)

# input bit is considered as an additional state (MSb) in this LUT, thus it has (states * 2) values
OUTPUT_LUT = [bits.from_int((xor_reduce_poly(x, G0) << 1) | xor_reduce_poly(x, G1), bits=2)
              for x in range(2 ** K)]


def do(data: bits) -> bits:
    """ See Figure 17-8—Convolutional encoder """
    output = bits('')
    shr = bits('0' * (K - 1))
    for bit in data:
        i = bit + shr
        output += OUTPUT_LUT[i.astype(int)]
        shr = i[:-1]  # advance the shift register

    return output


@njit(cache=True)
def trellis_kernel(rx):  # pragma: no cover
    """
    Each node in the trellis has 2 static parents (inputs from previous stage), also outputs are static. This LUT
    provides the parent IDs and the output values. Format: (parent1 id, parent1 output, parent2_id, parent2_output)

    For example, consider the data for the 4. trellis node:
        BUTTERFLY_LUT[4] -> (6, 2, 7, 1)
        Meaning, the first parent is trellis node 6 with state output of 2, and the second parent is node 7 with output 1.

    """
    BUTTERFLY_LUT = [(0, 0, 1, 3), (2, 2, 3, 1), (4, 0, 5, 3), (6, 2, 7, 1),
                     (8, 3, 9, 0), (10, 1, 11, 2), (12, 3, 13, 0), (14, 1, 15, 2),
                     (16, 3, 17, 0), (18, 1, 19, 2), (20, 3, 21, 0), (22, 1, 23, 2),
                     (24, 0, 25, 3), (26, 2, 27, 1), (28, 0, 29, 3), (30, 2, 31, 1),
                     (32, 1, 33, 2), (34, 3, 35, 0), (36, 1, 37, 2), (38, 3, 39, 0),
                     (40, 2, 41, 1), (42, 0, 43, 3), (44, 2, 45, 1), (46, 0, 47, 3),
                     (48, 2, 49, 1), (50, 0, 51, 3), (52, 2, 53, 1), (54, 0, 55, 3),
                     (56, 1, 57, 2), (58, 3, 59, 0), (60, 1, 61, 2), (62, 3, 63, 0),
                     (0, 3, 1, 0), (2, 1, 3, 2), (4, 3, 5, 0), (6, 1, 7, 2),
                     (8, 0, 9, 3), (10, 2, 11, 1), (12, 0, 13, 3), (14, 2, 15, 1),
                     (16, 0, 17, 3), (18, 2, 19, 1), (20, 0, 21, 3), (22, 2, 23, 1),
                     (24, 3, 25, 0), (26, 1, 27, 2), (28, 3, 29, 0), (30, 1, 31, 2),
                     (32, 2, 33, 1), (34, 0, 35, 3), (36, 2, 37, 1), (38, 0, 39, 3),
                     (40, 1, 41, 2), (42, 3, 43, 0), (44, 1, 45, 2), (46, 3, 47, 0),
                     (48, 1, 49, 2), (50, 3, 51, 0), (52, 1, 53, 2), (54, 3, 55, 0),
                     (56, 2, 57, 1), (58, 0, 59, 3), (60, 2, 61, 1), (62, 0, 63, 3)]

    """ Error between parent output and actual_input 
    error = ERROR_LUT[parent][actual_input]
    Actual input may contain depunctured stuff.
    """
    ERROR_LUT = [[0, 1, 1, 2, 0, 1, 0, 1],
                 [1, 0, 2, 1, 0, 1, 1, 0],
                 [1, 2, 0, 1, 1, 0, 0, 1],
                 [2, 1, 1, 0, 1, 0, 1, 0]]

    # compute the whole trellis map
    trellis = np.empty(shape=(len(rx) + 1, 64, 2), dtype=np.int64)
    trellis[0][0][0] = 0  # cost for the first node
    trellis[0, 1:, 0] = 1000  # set high cost for all starting nodes except the first one

    for i, actual_input in enumerate(rx):
        for j in range(STATES):
            parent1, parent1_out, parent2, parent2_out = BUTTERFLY_LUT[j]

            parent1_score = trellis[i][parent1][0] + ERROR_LUT[parent1_out][actual_input]
            parent2_score = trellis[i][parent2][0] + ERROR_LUT[parent2_out][actual_input]

            if parent1_score < parent2_score:
                trellis[i + 1][j][0] = parent1_score
                trellis[i + 1][j][1] = parent1
            else:
                trellis[i + 1][j][0] = parent2_score
                trellis[i + 1][j][1] = parent2

    # at this point we know the solution, but need to extract it by following the best node backwards (i.e. traversing as linked list)
    best_score = np.min(trellis[-1, :, 0])
    best_score_index = np.argmin(trellis[-1, :, 0])
    bits = np.zeros(shape=(len(rx)), dtype=np.int64)
    for i in range(len(trellis) - 1, 0, -1):
        next = trellis[i][best_score_index][1]
        if next > best_score_index or (next == 0 and best_score_index == 0):
            bits[i - 1] = 0
        else:
            bits[i - 1] = 1
        best_score_index = next

    return bits, best_score


def undo(data: bits) -> bits:
    LUT = {'00': 0, '01': 1, '10': 2, '11': 3, '0?': 4, '1?': 5, '?0': 6, '?1': 7}
    assume(is_divisible(data, by=2))
    data = [LUT[state_transition] for state_transition in data.split(2)]

    out, error_score = trellis_kernel(data)
    logger.debug(f'{len(out)//8}B, error_score={int(error_score)}')
    return bits(out)


def test_signal():
    """
    Uses 1/2 coding rate for signal field.
    """

    # IEEE Std 802.11-2016: Table I-7—Bit assignment for SIGNAL field
    input = bits('101100010011000000000000')

    # IEEE Std 802.11-2016: Table I-8—SIGNAL field bits after encoding
    expected = bits('110100011010000100000010001111100111000000000000')
    output = do(input)
    assert output == expected

    # test decoding
    decoded = undo(output)
    assert decoded == input

    # test decoding with bit errors
    # 4 errors
    output = bits('010100011010100101000010001111100111000000100000')
    decoded = undo(output)
    assert decoded == input


def test_i161():
    """
    Uses 3/4 coding rate for data field.
    """

    # Table I-15—The DATA bits after scrambling (and resetting tail bits)
    input = bits('0110110000011001100010011000111101101000001000011111010010100101011000010100111111010111101011100'
                 '0100100000011001111001100111010111001001011110001010011100110001100000000011110001101011011001111'
                 '100011111110000010010101100000110101100010010100110101001100111111111011110000010000010010101110'
                 '001111010100110001110010000011010000011011111000111001001010000110011001000100011001101100110111'
                 '110110101000111101100000001101110101001000000100111011001011111101111111000011010110001111011111'
                 '000110010100101110101101110000100011111001111001101010100100001000000111111101011111001010100111'
                 '01000101010101000100100000010001110100110110011110100100111011110011011001001110001101011110110'
                 '111110001110000000000100010000010011001101000010111110110001010001001110001011100111001000101011'
                 '01000001110110010010101000101101001000100010000000000001101110001111111000011101111001011001001')

    # Table I-16—The BCC encoded DATA bits
    expect = bits('0010101100001000101000011111000010011101101101011001101000011101010010101111101111101000110000101'
                  '0001111110000001100100001110011110000000100001111100000000110011110000011010011111010111011001010'
                  '10111110011000111111010101100100001111100010110110100101100110000011001010101011011001000100000101'
                  '01101000101110100110010000000110010010110011001000011001111010001110100100011100000100000101101101'
                  '11101101111100010111011000100000000010111110100010110111010110111100101011100101110110000111011001'
                  '11011101000011010001001001110110001001110000001001001100100100101011110000010010010010110110101011'
                  '1101110111000000100011001001111000111000000001101101001101011011000011011010100110000001001101010'
                  '0101111001011010100011001110010110000100000001110101110001001001101000001010001000000011011001001'
                  '11100010110001010001010010110111110001101001000100010001010110000110111111011011111001011110111110'
                  '01101111100010100011101111110010100101010100000011111111010010011010010001001110111001010111101100'
                  '01011011001000111001100101011111001010000011111011010100111010011111011110111000000100110111010110'
                  '001110111100101010000000011011011011001110100100000111010111011011000010111111')

    output = do(input)
    output = puncturer.do(output, coding_rate='3/4')
    assert output == expect

    # test decoding
    data = puncturer.undo(output, coding_rate='3/4')
    decoded = undo(data)
    assert decoded == input


@settings(deadline=None)
@given(binary())
# @given(binary(), sampled_from(['1/2', '2/3', '3/4']))
def test_hypothesis(data):
    data = bits(data)
    assert undo(do(data)) == data
