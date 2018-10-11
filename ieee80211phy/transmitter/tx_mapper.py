import numpy as np


def to_unsigned_int(data):
    return int(''.join(str(int(x)) for x in data), 2)


def int_to_bits(int, num_of_bits):
    bitstr = bin(int)[2:].zfill(num_of_bits)
    ret = [0 if x == '0' else 1 for x in bitstr]
    return ret


def mapper(bits, bits_per_symbol=1, normalize=True):
    BPSK_LUT = np.array([-1., 1.])
    QPSK_LUT = np.array([-1., 1.])
    QAM16_LUT = np.array([-3., -1., 3., 1.])
    QAM64_LUT = np.array([-7., -5., -3., -1., 1., 3., 5., 7.])  # BUG: NOT GRAY CODED
    if normalize:
        QPSK_LUT /= np.sqrt(2)
        QAM16_LUT /= np.sqrt(10)
        QAM64_LUT /= np.sqrt(42)

    bits = np.squeeze(np.reshape(bits, (-1, bits_per_symbol)))
    out = []
    for x in bits:
        if bits_per_symbol == 1:
            symbol = BPSK_LUT[x] + 0.0j
        elif bits_per_symbol == 2:
            symbol = QPSK_LUT[x[0]] + QPSK_LUT[x[1]] * 1j
        elif bits_per_symbol == 4:
            i_index = to_unsigned_int(x[0:2])
            q_index = to_unsigned_int(x[2:4])
            symbol = QAM16_LUT[i_index] + QAM16_LUT[q_index] * 1j
        elif bits_per_symbol == 6:
            i_index = to_unsigned_int(x[0:3])
            q_index = to_unsigned_int(x[3:6])
            symbol = QAM64_LUT[i_index] + QAM64_LUT[q_index] * 1j

        out.append(symbol)
    return out


def inset_pilots_and_pad(symbols):
    """ Inserts 4 pilots at position -21, -7, 7, 21 and pads start and end with zeroes
    Also adds a zero for the zero carrier
    """
    # TODO: simplified to always use 1+0j pilots!
    pad_head = [0 + 0j] * 6
    pad_tail = [0 + 0j] * 5
    symbols = pad_head + list(symbols) + pad_tail
    symbols.insert(11, 1 + 0j)  # pilot at carrier -21
    symbols.insert(25, 1 + 0j)  # pilot at carrier -7
    symbols.insert(32, 0 + 0j)  # zero at carrier 0
    symbols.insert(39, 1 + 0j)  # pilot at carrier 7
    symbols.insert(53, -1 + 0j)  # pilot at carrier 21

    return np.array(symbols)


def test_inset_pilots_and_pad():
    input = [(-0.316 + 0.316j), (-0.316 + 0.316j), (0.316 + 0.316j), (-0.949 - 0.949j), (0.316 + 0.949j),
             (0.316 + 0.316j), (0.316 - 0.949j), (-0.316 - 0.949j), (-0.316 + 0.316j), (-0.949 + 0.316j),
             (-0.949 - 0.949j), (-0.949 - 0.949j), (0.949 + 0.316j), (0.316 + 0.316j), (-0.949 - 0.316j),
             (-0.949 - 0.316j), (-0.949 - 0.316j), (-0.949 - 0.949j), (0.949 - 0.316j), (0.949 + 0.949j),
             (-0.949 - 0.316j), (0.316 - 0.316j), (-0.949 - 0.316j), (-0.949 + 0.949j), (-0.316 + 0.949j),
             (0.316 + 0.949j), (-0.949 + 0.316j), (0.949 - 0.949j), (0.316 + 0.316j), (-0.316 - 0.316j),
             (-0.316 + 0.949j), (0.949 - 0.316j), (-0.949 - 0.316j), (0.949 + 0.316j), (-0.316 + 0.949j),
             (0.949 + 0.316j), (0.949 - 0.316j), (0.949 - 0.949j), (-0.316 - 0.949j), (-0.949 + 0.316j),
             (-0.949 - 0.949j), (-0.949 - 0.949j), (-0.949 - 0.949j), (0.316 - 0.316j), (0.949 + 0.316j),
             (-0.949 + 0.316j), (-0.316 + 0.949j), (0.316 - 0.316j)]
    simplifed_expected = [0j, 0j, 0j, 0j, 0j, 0j, (-0.316 + 0.316j), (-0.316 + 0.316j), (0.316 + 0.316j), (-0.949 - 0.949j),
                (0.316 + 0.949j), (1 + 0j), (0.316 + 0.316j), (0.316 - 0.949j), (-0.316 - 0.949j), (-0.316 + 0.316j),
                (-0.949 + 0.316j), (-0.949 - 0.949j), (-0.949 - 0.949j), (0.949 + 0.316j), (0.316 + 0.316j),
                (-0.949 - 0.316j), (-0.949 - 0.316j), (-0.949 - 0.316j), (-0.949 - 0.949j), (1 + 0j), (0.949 - 0.316j),
                (0.949 + 0.949j), (-0.949 - 0.316j), (0.316 - 0.316j), (-0.949 - 0.316j), (-0.949 + 0.949j), 0j,
                (-0.316 + 0.949j), (0.316 + 0.949j), (-0.949 + 0.316j), (0.949 - 0.949j), (0.316 + 0.316j),
                (-0.316 - 0.316j), (1 + 0j), (-0.316 + 0.949j), (0.949 - 0.316j), (-0.949 - 0.316j), (0.949 + 0.316j),
                (-0.316 + 0.949j), (0.949 + 0.316j), (0.949 - 0.316j), (0.949 - 0.949j), (-0.316 - 0.949j),
                (-0.949 + 0.316j), (-0.949 - 0.949j), (-0.949 - 0.949j), (-0.949 - 0.949j), (1 + 0j), (0.316 - 0.316j),
                (0.949 + 0.316j), (-0.949 + 0.316j), (-0.316 + 0.949j), (0.316 - 0.316j), 0j, 0j, 0j, 0j, 0j]

    expected = [0j, 0j, 0j, 0j, 0j, 0j, (-0.316 + 0.316j), (-0.316 + 0.316j), (0.316 + 0.316j), (-0.949 - 0.949j),
                (0.316 + 0.949j), (1 + 0j), (0.316 + 0.316j), (0.316 - 0.949j), (-0.316 - 0.949j), (-0.316 + 0.316j),
                (-0.949 + 0.316j), (-0.949 - 0.949j), (-0.949 - 0.949j), (0.949 + 0.316j), (0.316 + 0.316j),
                (-0.949 - 0.316j), (-0.949 - 0.316j), (-0.949 - 0.316j), (-0.949 - 0.949j), (1 + 0j), (0.949 - 0.316j),
                (0.949 + 0.949j), (-0.949 - 0.316j), (0.316 - 0.316j), (-0.949 - 0.316j), (-0.949 + 0.949j), 0j,
                (-0.316 + 0.949j), (0.316 + 0.949j), (-0.949 + 0.316j), (0.949 - 0.949j), (0.316 + 0.316j),
                (-0.316 - 0.316j), (1 + 0j), (-0.316 + 0.949j), (0.949 - 0.316j), (-0.949 - 0.316j), (0.949 + 0.316j),
                (-0.316 + 0.949j), (0.949 + 0.316j), (0.949 - 0.316j), (0.949 - 0.949j), (-0.316 - 0.949j),
                (-0.949 + 0.316j), (-0.949 - 0.949j), (-0.949 - 0.949j), (-0.949 - 0.949j), (-1 + 0j), (0.316 - 0.316j),
                (0.949 + 0.316j), (-0.949 + 0.316j), (-0.316 + 0.949j), (0.316 - 0.316j), 0j, 0j, 0j, 0j, 0j]

    output = inset_pilots_and_pad(input)
    np.testing.assert_equal(expected, np.round(output, 3))


def test_mapper_bpsk():
    bits = np.array([0, 1, 0, 1])
    symbols = mapper(bits, bits_per_symbol=1)
    assert symbols == [(-1 + 0j), (1 + 0j), (-1 + 0j), (1 + 0j)]


def test_mapper_qpsk():
    bits = np.array([0, 0, 0, 1, 1, 0, 1, 1])
    symbols = mapper(bits, bits_per_symbol=2, normalize=False)
    assert symbols == [(-1 - 1j), (-1 + 1j), (1 - 1j), (1 + 1j)]


def test_mapper_qam16():
    bits_per_symbol = 4
    bits = np.array([int_to_bits(x, bits_per_symbol) for x in range(16)]).flatten()
    symbols = mapper(bits, bits_per_symbol, normalize=False)
    assert symbols == [(-3 - 3j), (-3 - 1j), (-3 + 1j), (-3 + 3j),
                       (-1 - 3j), (-1 - 1j), (-1 + 1j), (-1 + 3j),
                       (1 - 3j), (1 - 1j), (1 + 1j), (1 + 3j),
                       (3 - 3j), (3 - 1j), (3 + 1j), (3 + 3j)]


def test_mapper_qam64():
    bits_per_symbol = 6
    bits = np.array([int_to_bits(x, bits_per_symbol) for x in range(64)]).flatten()
    symbols = mapper(bits, bits_per_symbol, normalize=False)
    assert symbols == [(-7 - 7j), (-7 - 5j), (-7 - 3j), (-7 - 1j), (-7 + 1j), (-7 + 3j), (-7 + 5j), (-7 + 7j),
                       (-5 - 7j), (-5 - 5j), (-5 - 3j), (-5 - 1j), (-5 + 1j), (-5 + 3j), (-5 + 5j), (-5 + 7j),
                       (-3 - 7j), (-3 - 5j), (-3 - 3j), (-3 - 1j), (-3 + 1j), (-3 + 3j), (-3 + 5j), (-3 + 7j),
                       (-1 - 7j), (-1 - 5j), (-1 - 3j), (-1 - 1j), (-1 + 1j), (-1 + 3j), (-1 + 5j), (-1 + 7j),
                       (1 - 7j), (1 - 5j), (1 - 3j), (1 - 1j), (1 + 1j), (1 + 3j), (1 + 5j), (1 + 7j),
                       (3 - 7j), (3 - 5j), (3 - 3j), (3 - 1j), (3 + 1j), (3 + 3j), (3 + 5j), (3 + 7j),
                       (5 - 7j), (5 - 5j), (5 - 3j), (5 - 1j), (5 + 1j), (5 + 3j), (5 + 5j), (5 + 7j),
                       (7 - 7j), (7 - 5j), (7 - 3j), (7 - 1j), (7 + 1j), (7 + 3j), (7 + 5j), (7 + 7j)]


def test_iee_mapper():
    """ IEEE Std 802.11-2016 - Table I-19—Interleaved bits of first DATA symbol """
    input = '011101111111000011101111110001000111001100000000101111110001000100010000100110100001110100010010011011100011100011110101011010010001101101101011100110000100001100000000000011011011001101101101'
    input = np.array([1 if x == '1' else 0 for x in input])
    symbols = mapper(input, bits_per_symbol=4)

    # expected = [0j, 0j, 0j, 0j, 0j, 0j, (-0.316+0.316j), (-0.316+0.316j), (0.316+0.316j), (-0.949-0.949j), (0.316+0.949j), (1+0j), (0.316+0.316j), (0.316-0.949j), (-0.316-0.949j), (-0.316+0.316j), (-0.949+0.316j), (-0.949-0.949j), (-0.949-0.949j), (0.949+0.316j), (0.316+0.316j), (-0.949-0.316j), (-0.949-0.316j), (-0.949-0.316j), (-0.949-0.949j), (1+0j), (0.949-0.316j), (0.949+0.949j), (-0.949-0.316j), (0.316-0.316j), (-0.949-0.316j), (-0.949+0.949j), 0j, (-0.316+0.949j), (0.316+0.949j), (-0.949+0.316j), (0.949-0.949j), (0.316+0.316j), (-0.316-0.316j), (1+0j), (-0.316+0.949j), (0.949-0.316j), (-0.949-0.316j), (0.949+0.316j), (-0.316+0.949j), (0.949+0.316j), (0.949-0.316j), (0.949-0.949j), (-0.316-0.949j), (-0.949+0.316j), (-0.949-0.949j), (-0.949-0.949j), (-0.949-0.949j), (-1+0j), (0.316-0.316j), (0.949+0.316j), (-0.949+0.316j), (-0.316+0.949j), (0.316-0.316j), 0j, 0j, 0j, 0j, 0j]
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
    pass


def test_iee_signal():
    # Table I-9—SIGNAL field bits after interleaving
    input = '100101001101000000010100100000110010010010010100'
    input = np.array([1 if x == '1' else 0 for x in input])
    symbols = mapper(input, bits_per_symbol=1)
    symbols = inset_pilots_and_pad(symbols)

    # Table I-10—Frequency domain representation of SIGNAL field
    expected = [0j, 0j, 0j, 0j, 0j, 0j, (1+0j), (-1+0j), (-1+0j), (1+0j), (-1+0j), (1+0j), (1+0j), (-1+0j), (-1+0j), (1+0j), (1+0j), (-1+0j), (1+0j), (-1+0j), (-1+0j), (-1+0j), (-1+0j), (-1+0j), (-1+0j), (1+0j), (-1+0j), (1+0j), (-1+0j), (1+0j), (-1+0j), (-1+0j), 0j, (1+0j), (-1+0j), (-1+0j), (-1+0j), (-1+0j), (-1+0j), (1+0j), (1+0j), (1+0j), (-1+0j), (-1+0j), (1+0j), (-1+0j), (-1+0j), (1+0j), (-1+0j), (-1+0j), (1+0j), (-1+0j), (-1+0j), (-1+0j), (1+0j), (-1+0j), (1+0j), (-1+0j), (-1+0j), 0j, 0j, 0j, 0j, 0j]
    np.testing.assert_equal(expected, np.round(symbols, 3))

