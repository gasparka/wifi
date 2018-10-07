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
    QAM64_LUT = np.array([-7., -5., -3., -1., 1., 3., 5., 7.]) # BUG: NOT GRAY CODED
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
    assert symbols == [(-7-7j), (-7-5j), (-7-3j), (-7-1j), (-7+1j), (-7+3j), (-7+5j), (-7+7j),
                       (-5-7j), (-5-5j), (-5-3j), (-5-1j), (-5+1j), (-5+3j), (-5+5j), (-5+7j),
                       (-3-7j), (-3-5j), (-3-3j), (-3-1j), (-3+1j), (-3+3j), (-3+5j), (-3+7j),
                       (-1-7j), (-1-5j), (-1-3j), (-1-1j), (-1+1j), (-1+3j), (-1+5j), (-1+7j),
                       (1-7j), (1-5j), (1-3j), (1-1j), (1+1j), (1+3j), (1+5j), (1+7j),
                       (3-7j), (3-5j), (3-3j), (3-1j), (3+1j), (3+3j), (3+5j), (3+7j),
                       (5-7j), (5-5j), (5-3j), (5-1j), (5+1j), (5+3j), (5+5j), (5+7j),
                       (7-7j), (7-5j), (7-3j), (7-1j), (7+1j), (7+3j), (7+5j), (7+7j)]


def test_iee_mapper():
    """ IEEE Std 802.11-2016 - Table I-19—Interleaved bits of first DATA symbol """
    input = '011101111111000011101111110001000111001100000000101111110001000100010000100110100001110100010010011011100011100011110101011010010001101101101011100110000100001100000000000011011011001101101101'
    input = np.array([1 if x == '1' else 0 for x in input])
    symbols = mapper(input, bits_per_symbol=4)

    # expected = [0j, 0j, 0j, 0j, 0j, 0j, (-0.316+0.316j), (-0.316+0.316j), (0.316+0.316j), (-0.949-0.949j), (0.316+0.949j), (1+0j), (0.316+0.316j), (0.316-0.949j), (-0.316-0.949j), (-0.316+0.316j), (-0.949+0.316j), (-0.949-0.949j), (-0.949-0.949j), (0.949+0.316j), (0.316+0.316j), (-0.949-0.316j), (-0.949-0.316j), (-0.949-0.316j), (-0.949-0.949j), (1+0j), (0.949-0.316j), (0.949+0.949j), (-0.949-0.316j), (0.316-0.316j), (-0.949-0.316j), (-0.949+0.949j), 0j, (-0.316+0.949j), (0.316+0.949j), (-0.949+0.316j), (0.949-0.949j), (0.316+0.316j), (-0.316-0.316j), (1+0j), (-0.316+0.949j), (0.949-0.316j), (-0.949-0.316j), (0.949+0.316j), (-0.316+0.949j), (0.949+0.316j), (0.949-0.316j), (0.949-0.949j), (-0.316-0.949j), (-0.949+0.316j), (-0.949-0.949j), (-0.949-0.949j), (-0.949-0.949j), (-1+0j), (0.316-0.316j), (0.949+0.316j), (-0.949+0.316j), (-0.316+0.949j), (0.316-0.316j), 0j, 0j, 0j, 0j, 0j]
    expected = [(-0.316+0.316j), (-0.316+0.316j), (0.316+0.316j), (-0.949-0.949j), (0.316+0.949j), (0.316+0.316j), (0.316-0.949j), (-0.316-0.949j), (-0.316+0.316j), (-0.949+0.316j), (-0.949-0.949j), (-0.949-0.949j), (0.949+0.316j), (0.316+0.316j), (-0.949-0.316j), (-0.949-0.316j), (-0.949-0.316j), (-0.949-0.949j), (0.949-0.316j), (0.949+0.949j), (-0.949-0.316j), (0.316-0.316j), (-0.949-0.316j), (-0.949+0.949j), (-0.316+0.949j), (0.316+0.949j), (-0.949+0.316j), (0.949-0.949j), (0.316+0.316j), (-0.316-0.316j), (-0.316+0.949j), (0.949-0.316j), (-0.949-0.316j), (0.949+0.316j), (-0.316+0.949j), (0.949+0.316j), (0.949-0.316j), (0.949-0.949j), (-0.316-0.949j), (-0.949+0.316j), (-0.949-0.949j), (-0.949-0.949j), (-0.949-0.949j), (0.316-0.316j), (0.949+0.316j), (-0.949+0.316j), (-0.316+0.949j), (0.316-0.316j)]

    np.testing.assert_equal(expected, np.round(symbols,3))
    pass