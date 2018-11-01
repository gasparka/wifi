import numpy as np

from ieee80211phy.transmitter.subcarrier_modulation_mapping import mapper

PILOT_POLARITY = [1, 1, 1, 1, -1, -1, -1, 1, -1, -1, -1, -1, 1, 1, -1, 1, -1, -1, 1, 1, -1, 1, 1, -1, 1, 1, 1, 1, 1, 1,
                  -1, 1, 1, 1, -1, 1, 1, -1, -1, 1, 1, 1, -1, 1, -1, -1, -1, 1, -1, 1, -1, -1, 1, -1, -1, 1, 1, 1, 1, 1,
                  -1, -1,
                  1, 1, -1, -1, 1, -1, 1, -1, 1, 1, -1, -1, -1, 1, 1, -1, -1, -1, -1, 1, -1, -1, 1, -1, 1, 1, 1, 1, -1, 1,
                  -1,
                  1, -1, 1, -1, -1, -1, -1, -1, 1, -1, 1, 1, -1, 1, -1, 1, 1, 1, -1, -1, 1, -1, -1, -1, 1, 1, 1, -1, -1,
                  -1, -1, -1,
                  -1, -1]

def ifft_guard(symbols):
    """
        l) For each group of subcarriers -26 to 26, convert the subcarriers to time domain using inverse
        Fourier transform. Prepend to the Fourier-transformed waveform a circular extension of itself thus
        forming a GI, and truncate the resulting periodic waveform to a single OFDM symbol length by
        applying time domain windowing. Refer to 17.3.5.10 for details.

        Note: i am doing time domain windowing later, makes no sense to have it here
    """
    ifft = np.fft.ifft(symbols)
    result = np.concatenate([ifft[48:], ifft])
    return result


def map_to_carriers(symbol):
    """
    j) Divide the complex number string into groups of 48 complex numbers. Each such group is
        associated with one OFDM symbol. In each group, the complex numbers are numbered 0 to 47 and
        mapped hereafter into OFDM subcarriers numbered -26 to -22, -20 to -8, -6 to -1, 1 to 6, 8 to 20,
        and 22 to 26. The subcarriers -21, -7, 7, and 21 are skipped and, subsequently, used for inserting
        pilot subcarriers. The 0 subcarrier, associated with center frequency, is omitted and filled with the
        value 0. Refer to 17.3.5.10 for details.

        Clarification: IFFT of 64 points are used. 802.11 uses 48 subcarriers + 4 pilot tones, thus leaving
        12 empty tones. This is solved by setting -32..-27, 0 and 27..31 subcarriers to 0+0j.
    """
    carrier = np.zeros(64, dtype=np.complex64)
    carrier[-32] = 0
    carrier[-31] = 0
    carrier[-30] = 0
    carrier[-29] = 0
    carrier[-28] = 0
    carrier[-27] = 0
    carrier[-26] = symbol[0]
    carrier[-25] = symbol[1]
    carrier[-24] = symbol[2]
    carrier[-23] = symbol[3]
    carrier[-22] = symbol[4]
    carrier[-21] = 0
    carrier[-20] = symbol[5]
    carrier[-19] = symbol[6]
    carrier[-18] = symbol[7]
    carrier[-17] = symbol[8]
    carrier[-16] = symbol[9]
    carrier[-15] = symbol[10]
    carrier[-14] = symbol[11]
    carrier[-13] = symbol[12]
    carrier[-12] = symbol[13]
    carrier[-11] = symbol[14]
    carrier[-10] = symbol[15]
    carrier[-9] = symbol[16]
    carrier[-8] = symbol[17]
    carrier[-7] = 0
    carrier[-6] = symbol[18]
    carrier[-5] = symbol[19]
    carrier[-4] = symbol[20]
    carrier[-3] = symbol[21]
    carrier[-2] = symbol[22]
    carrier[-1] = symbol[23]
    carrier[0] = 0
    carrier[1] = symbol[24]
    carrier[2] = symbol[25]
    carrier[3] = symbol[26]
    carrier[4] = symbol[27]
    carrier[5] = symbol[28]
    carrier[6] = symbol[29]
    carrier[7] = 0
    carrier[8] = symbol[30]
    carrier[9] = symbol[31]
    carrier[10] = symbol[32]
    carrier[11] = symbol[33]
    carrier[12] = symbol[34]
    carrier[13] = symbol[35]
    carrier[14] = symbol[36]
    carrier[15] = symbol[37]
    carrier[16] = symbol[38]
    carrier[17] = symbol[39]
    carrier[18] = symbol[40]
    carrier[19] = symbol[41]
    carrier[20] = symbol[42]
    carrier[21] = 0
    carrier[22] = symbol[43]
    carrier[23] = symbol[44]
    carrier[24] = symbol[45]
    carrier[25] = symbol[46]
    carrier[26] = symbol[47]
    carrier[27] = 0
    carrier[28] = 0
    carrier[29] = 0
    carrier[30] = 0
    carrier[31] = 0

    return carrier


def demap_from_carriers(carrier):
    symbol = [carrier[-26], carrier[-25], carrier[-24], carrier[-23], carrier[-22], carrier[-20], carrier[-19],
              carrier[-18], carrier[-17], carrier[-16], carrier[-15], carrier[-14], carrier[-13], carrier[-12],
              carrier[-11], carrier[-10], carrier[-9], carrier[-8], carrier[-6], carrier[-5], carrier[-4], carrier[-3],
              carrier[-2], carrier[-1], carrier[1], carrier[2], carrier[3], carrier[4], carrier[5], carrier[6],
              carrier[8], carrier[9], carrier[10], carrier[11], carrier[12], carrier[13], carrier[14], carrier[15],
              carrier[16], carrier[17], carrier[18], carrier[19], carrier[20], carrier[22], carrier[23], carrier[24],
              carrier[25], carrier[26]]

    return symbol


def get_derotated_pilots(carrier, symbol_number):
    pilots = np.array([carrier[-21], carrier[-7], carrier[7], -carrier[21]]) * PILOT_POLARITY[symbol_number % 127]
    return pilots


def insert_pilots(ofdm_symbol, i):
    """
        k) Four subcarriers are inserted as pilots into positions -21, -7, 7, and 21. The total number of the
    subcarriers is 52 (48 + 4). Refer to 17.3.5.9 for details.

    In each OFDM symbol, four of the subcarriers are dedicated to pilot signals in order to make the coherent
    detection robust against frequency offsets and phase noise. These pilot signals shall be put in subcarriers
    -21, -7, 7, and 21. The pilots shall be BPSK modulated by a pseudo-binary sequence to prevent the
    generation of spectral lines. The contribution of the pilot subcarriers to each OFDM symbol is described in
    17.3.5.10.
    
    :param ofdm_symbol:
    :param i: symbol position, note that SIGNAL field is pos 0
    :return:
    """

    pilots = np.array([1, 1, 1, -1]) * PILOT_POLARITY[i % 127]
    ofdm_symbol[-21] = pilots[0]
    ofdm_symbol[-7] = pilots[1]
    ofdm_symbol[7] = pilots[2]
    ofdm_symbol[21] = pilots[3]
    return ofdm_symbol


def test_ofdm_i18():
    # IEEE Std 802.11-2016 - Table I-19—Interleaved bits of first DATA symbol
    input = '0111011111110000111011111100010001110011000000001011111100010001000100001001101000011101000100100110111' \
            '00011100011110101011010010001101101101011100110000100001100000000000011011011001101101101 '
    symbols = mapper(input, bits_per_symbol=4)
    output = map_to_carriers(symbols)
    output = insert_pilots(output, 0)
    output = ifft_guard(output)

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

    output = np.round(output, 3)
    np.testing.assert_equal(expected[1:-1], output[
                                            1:-1])  # skipping first and last as they are involved in time windowing - which i will perform later
