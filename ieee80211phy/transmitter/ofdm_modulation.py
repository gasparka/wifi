import numpy as np

from ieee80211phy.transmitter.subcarrier_modulation_mapping import mapper


def insert_pilots_and_pad(symbols):
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


def ifft_guard_window(symbols):
    """ j) Divide the complex number string into groups of 48 complex numbers. Each such group is
        associated with one OFDM symbol. In each group, the complex numbers are numbered 0 to 47 and
        mapped hereafter into OFDM subcarriers numbered –26 to –22, –20 to –8, –6 to –1, 1 to 6, 8 to 20,
        and 22 to 26. The subcarriers –21, –7, 7, and 21 are skipped and, subsequently, used for inserting
        pilot subcarriers. The 0 subcarrier, associated with center frequency, is omitted and filled with the
        value 0. Refer to 17.3.5.10 for details.

        k) Four subcarriers are inserted as pilots into positions –21, –7, 7, and 21. The total number of the
        subcarriers is 52 (48 + 4). Refer to 17.3.5.9 for details.

        l) For each group of subcarriers –26 to 26, convert the subcarriers to time domain using inverse
        Fourier transform. Prepend to the Fourier-transformed waveform a circular extension of itself thus
        forming a GI, and truncate the resulting periodic waveform to a single OFDM symbol length by
        applying time domain windowing. Refer to 17.3.5.10 for details.

        Clarification: IFFT of 64 points are used. 802.11 uses 48 subcarriers + 4 pilot tones, thus leaving
        12 empty tones. This is solved by setting -32..-27, 0 and 27..31 subcarriers to 0+0j.

    """
    symbols = insert_pilots_and_pad(symbols)
    symbols = np.fft.fftshift(symbols)
    ifft = np.fft.ifft(symbols)
    result = np.concatenate([ifft[48:], ifft])
    return result


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

    output = insert_pilots_and_pad(input)
    np.testing.assert_equal(expected, np.round(output, 3))


def test_ofdm_i18():

    # IEEE Std 802.11-2016 - Table I-19—Interleaved bits of first DATA symbol
    input = '0111011111110000111011111100010001110011000000001011111100010001000100001001101000011101000100100110111' \
            '00011100011110101011010010001101101101011100110000100001100000000000011011011001101101101 '
    symbols = mapper(input, bits_per_symbol=4)

    # Table I-25—Time domain representation of the DATA field: symbol 1of 6
    expected = [(-0.139 + 0.05j), (0.004 + 0.014j), (0.011 - 0.1j), (-0.097 - 0.02j), (0.062 + 0.081j),
                (0.124 + 0.139j), (0.104 - 0.015j), (0.173 - 0.14j), (-0.04 + 0.006j), (-0.133 + 0.009j),
                (-0.002 - 0.043j), (-0.047 + 0.092j), (-0.109 + 0.082j), (-0.024 + 0.01j), (0.096 + 0.019j),
                (0.019 - 0.023j), (-0.087 - 0.049j), (0.002 + 0.058j), (-0.021 + 0.228j), (-0.103 + 0.023j),
                (-0.019 - 0.175j), (0.018 + 0.132j), (-0.071 + 0.16j), (-0.153 - 0.062j), (-0.107 + 0.028j),
                (0.055 + 0.14j), (0.07 + 0.103j), (-0.056 + 0.025j), (-0.043 + 0.002j), (0.016 - 0.118j),
                (0.026 - 0.071j), (0.033 + 0.177j), (0.02 - 0.021j), (0.035 - 0.088j), (-0.008 + 0.101j),
                (-0.035 - 0.01j), (0.065 + 0.03j), (0.092 - 0.034j), (0.032 - 0.123j), (-0.018 + 0.092j), -0.006j,
                (-0.006 - 0.056j), (-0.019 + 0.04j), (0.053 - 0.131j), (0.022 - 0.133j), (0.104 - 0.032j),
                (0.163 - 0.045j), (-0.105 - 0.03j), (-0.11 - 0.069j), (-0.008 - 0.092j), (-0.049 - 0.043j),
                (0.085 - 0.017j), (0.09 + 0.063j), (0.015 + 0.153j), (0.049 + 0.094j), (0.011 + 0.034j),
                (-0.012 + 0.012j), (-0.015 - 0.017j), (-0.061 + 0.031j), (-0.07 - 0.04j), (0.011 - 0.109j),
                (0.037 - 0.06j), (-0.003 - 0.178j), (-0.007 - 0.128j), (-0.059 + 0.1j), (0.004 + 0.014j),
                (0.011 - 0.1j), (-0.097 - 0.02j), (0.062 + 0.081j), (0.124 + 0.139j), (0.104 - 0.015j),
                (0.173 - 0.14j), (-0.04 + 0.006j), (-0.133 + 0.009j), (-0.002 - 0.043j), (-0.047 + 0.092j),
                (-0.109 + 0.082j), (-0.024 + 0.01j), (0.096 + 0.019j), (0.019 - 0.023j)]

    output = np.round(ifft_guard_window(symbols), 3)
    np.testing.assert_equal(expected[1:-1], output[1:-1]) # skipping first and last as they are involved in time windowing - which i will perform later

