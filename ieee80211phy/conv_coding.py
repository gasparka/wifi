import logging
from textwrap import wrap
import numpy as np
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


def conv_decode(rx, coding_rate='1/2'):
    """ See 'Bits, Signals, and Packets: An Introduction to Digital Communications and Networks' ->
        'Viterbi Decoding of Convolutional Codes (PDF - 1.4MB)'
    """

    def butterfly(state, expected, scores):
        def error(acutal, expected):
            ret = 0
            if acutal[0] != expected[0] and expected[0] != 'X':
                ret += 1

            if acutal[1] != expected[1] and expected[1] != 'X':
                ret += 1

            return ret

        input_bit = (state << 1) >> (K - 1)  # 0 or 1

        parent1 = (state << 1) % STATES
        parent1_out = int_to_binstr(OUTPUT_LUT[(input_bit * STATES) | parent1], bits=2)
        parent1_error = error(parent1_out, expected)
        parent1_score = scores[parent1][0] + parent1_error

        parent2 = (parent1 + 1) % STATES
        parent2_out = int_to_binstr(OUTPUT_LUT[(input_bit * STATES) | parent2], bits=2)
        parent2_error = error(parent2_out, expected)
        parent2_score = scores[parent2][0] + parent2_error

        if parent1_score < parent2_score:
            return parent1_score, scores[parent1][1] + str(input_bit)
        else:
            return parent2_score, scores[parent2][1] + str(input_bit)

    rx = _puncture(rx, coding_rate, undo=True)
    scores = [(0, '')] + ([(1000, '')] * (STATES - 1)) # (state score, decoded bits)
    for expect in wrap(rx, 2):
        scores = [butterfly(i, expect, scores) for i in range(len(scores))]

    min_score_index = int(np.argmin([x[0] for x in scores]))
    bits = scores[min_score_index][1]
    logger.info(f'Decoded {len(bits)} bits, score={scores[min_score_index][0]}, rate={coding_rate}')
    return bits


def conv_encode(data, coding_rate='1/2'):
    """ See Figure 17-8—Convolutional encoder """
    output = ''
    shr = '0' * (K - 1)
    for bit in data:
        i = bit + shr
        output += int_to_binstr(OUTPUT_LUT[int(i, 2)], bits=2)
        shr = i[:-1]  # advance the shift register

    return _puncture(output, coding_rate)


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
