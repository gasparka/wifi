"""
17.3.5.7 Data interleaving
--------------------------

All encoded data bits shall be interleaved by a block interleaver with a block size corresponding to the
number of bits in a single OFDM symbol. The interleaver is defined by a two-step permutation.

"""
from typing import List
import numpy as np
from ieee80211phy.bits import bits


def first_permute(coded_bits_symbol: int) -> List[int]:
    """ The first permutation causes adjacent coded bits to be mapped onto nonadjacent subcarriers. """
    lut = [int((coded_bits_symbol / 16) * (k % 16) + np.floor(k / 16))
           for k in range(coded_bits_symbol)]
    return lut


def second_permute(coded_bits_symbol: int, coded_bits_subcarrier: int) -> List[int]:
    """ The second permutation causes adjacent coded bits to be mapped alternately onto less and more significant bits of the
    constellation and, thereby, long runs of low reliability (LSB) bits are avoided. """

    s = max(coded_bits_subcarrier / 2, 1)
    lut = [int(s * np.floor(i / s) + (i + coded_bits_symbol - np.floor((16 * i) / coded_bits_symbol)) % s)
           for i in range(coded_bits_symbol)]
    return lut


def inverse_permute(x: List[int]) -> List[int]:
    result = np.array(x)
    result[x] = np.arange(0, len(result))
    return result.tolist()


def apply(data: bits, coded_bits_symbol: int, coded_bits_subcarrier: int) -> bits:
    """
    Divide the encoded bit string into groups of NCBPS bits. Within each group, perform an
    “interleaving” (reordering) of the bits according to a rule corresponding to the TXVECTOR
    parameter RATE. Refer to 17.3.5.7 for details.
    """

    table = first_permute(coded_bits_symbol)
    table = inverse_permute(table)
    first_result = data[table]

    table = second_permute(coded_bits_symbol, coded_bits_subcarrier)
    table = inverse_permute(table)
    second_result = first_result[table]
    return second_result


def undo(data: bits, coded_bits_symbol: int, coded_bits_subcarrier: int) -> bits:
    table = second_permute(coded_bits_symbol, coded_bits_subcarrier)
    first_result = data[table]

    table = first_permute(coded_bits_symbol)
    second_result = first_result[table]
    return second_result


def test_first_permutation_table():
    result = first_permute(192)

    # IEEE Std 802.11-2016: Table I-17—First permutation
    expect = [0, 12, 24, 36, 48, 60, 72, 84, 96, 108, 120, 132, 144, 156, 168, 180, 1, 13, 25, 37, 49, 61, 73, 85, 97,
              109, 121, 133, 145, 157, 169, 181, 2, 14, 26, 38, 50, 62, 74, 86, 98, 110, 122, 134, 146, 158, 170, 182,
              3, 15, 27, 39, 51, 63, 75, 87, 99, 111, 123, 135, 147, 159, 171, 183, 4, 16, 28, 40, 52, 64, 76, 88, 100,
              112, 124, 136, 148, 160, 172, 184, 5, 17, 29, 41, 53, 65, 77, 89, 101, 113, 125, 137, 149, 161, 173, 185,
              6, 18, 30, 42, 54, 66, 78, 90, 102, 114, 126, 138, 150, 162, 174, 186, 7, 19, 31, 43, 55, 67, 79, 91, 103,
              115, 127, 139, 151, 163, 175, 187, 8, 20, 32, 44, 56, 68, 80, 92, 104, 116, 128, 140, 152, 164, 176, 188,
              9, 21, 33, 45, 57, 69, 81, 93, 105, 117, 129, 141, 153, 165, 177, 189, 10, 22, 34, 46, 58, 70, 82, 94,
              106, 118, 130, 142, 154, 166, 178, 190, 11, 23, 35, 47, 59, 71, 83, 95, 107, 119, 131, 143, 155, 167, 179,
              191]
    assert result == expect


def test_second_permutation_table():
    result = second_permute(192, 4)

    # IEEE Std 802.11-2016: Table I-18—Second permutation
    expect = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 13, 12, 15, 14, 17, 16, 19, 18, 21, 20, 23, 22, 24, 25, 26, 27, 28,
              29, 30, 31, 32, 33, 34, 35, 37, 36, 39, 38, 41, 40, 43, 42, 45, 44, 47, 46, 48, 49, 50, 51, 52, 53, 54,
              55, 56, 57, 58, 59, 61, 60, 63, 62, 65, 64, 67, 66, 69, 68, 71, 70, 72, 73, 74, 75, 76, 77, 78, 79, 80,
              81, 82, 83, 85, 84, 87, 86, 89, 88, 91, 90, 93, 92, 95, 94, 96, 97, 98, 99, 100, 101, 102, 103, 104, 105,
              106, 107, 109, 108, 111, 110, 113, 112, 115, 114, 117, 116, 119, 118, 120, 121, 122, 123, 124, 125, 126,
              127, 128, 129, 130, 131, 133, 132, 135, 134, 137, 136, 139, 138, 141, 140, 143, 142, 144, 145, 146, 147,
              148, 149, 150, 151, 152, 153, 154, 155, 157, 156, 159, 158, 161, 160, 163, 162, 165, 164, 167, 166, 168,
              169, 170, 171, 172, 173, 174, 175, 176, 177, 178, 179, 181, 180, 183, 182, 185, 184, 187, 186, 189, 188,
              191, 190]
    assert result == expect


def test_i143():
    # IEEE Std 802.11-2016: Table I-8—SIGNAL field bits after encoding
    input = bits('110100011010000100000010001111100111000000000000')

    # IEEE Std 802.11-2016: Table I-9—SIGNAL field bits after interleaving
    expect = bits('100101001101000000010100100000110010010010010100')
    result = apply(input, 48, 1)
    assert result == expect

    # test reverse
    result = undo(expect, 48, 1)
    assert result == input


def test_i162():
    # IEEE Std 802.11-2016: Table I-16—The BCC encoded DATA bits
    input = bits('001010110000100010100001111100001001110110110101100110100001110101001010111110111110'
                 '100011000010100011111100000011001000011100111100000001000011111000000001100111100000'
                 '110100111110101110110010')

    # IEEE Std 802.11-2016: Table I-19—Interleaved bits of first DATA symbol
    expect = bits('011101111111000011101111110001000111001100000000101111110001000100010000100110100001110'
                  '1000100100110111000111000111101010110100100011011011010111001100001000011000000000000110'
                  '11011001101101101')

    result = apply(input, 192, 4)
    assert result == expect

    # test reverse
    result = undo(expect, 192, 4)
    assert result == input
