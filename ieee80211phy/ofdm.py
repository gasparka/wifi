import queue
from collections import deque
from typing import Tuple
import numpy as np

PILOT_POLARITY = [1, 1, 1, 1, -1, -1, -1, 1, -1, -1, -1, -1, 1, 1, -1, 1, -1, -1, 1, 1, -1, 1, 1, -1, 1, 1, 1, 1, 1, 1,
                  -1, 1, 1, 1, -1, 1, 1, -1, -1, 1, 1, 1, -1, 1, -1, -1, -1, 1, -1, 1, -1, -1, 1, -1, -1, 1, 1, 1, 1, 1,
                  -1, -1, 1, 1, -1, -1, 1, -1, 1, -1, 1, 1, -1, -1, -1, 1, 1, -1, -1, -1, -1, 1, -1, -1, 1, -1, 1, 1, 1,
                  1, -1, 1, -1, 1, -1, 1, -1, -1, -1, -1, -1, 1, -1, 1, 1, -1, 1, -1, 1, 1, 1, -1, -1, 1, -1, -1, -1, 1,
                  1, 1, -1, -1, -1, -1, -1, -1, -1]


def modulate_ofdm(ofdm_symbol: np.ndarray, index_in_package: int) -> np.ndarray:
    """
    j) Divide the complex number string into groups of 48 complex numbers. Each such group is
        associated with one OFDM symbol. In each group, the complex numbers are numbered 0 to 47 and
        mapped hereafter into OFDM subcarriers numbered -26 to -22, -20 to -8, -6 to -1, 1 to 6, 8 to 20,
        and 22 to 26. The subcarriers -21, -7, 7, and 21 are skipped and, subsequently, used for inserting
        pilot subcarriers. The 0 subcarrier, associated with center frequency, is omitted and filled with the
        value 0. Refer to 17.3.5.10 for details.

        Clarification: IFFT of 64 points are used. 802.11 uses 48 subcarriers + 4 pilot tones, thus leaving
        12 empty tones. This is solved by setting -32..-27, 0 and 27..31 subcarriers to 0+0j.

    k) Four subcarriers are inserted as pilots into positions -21, -7, 7, and 21. The total number of the
        subcarriers is 52 (48 + 4). Refer to 17.3.5.9 for details.

        In each OFDM symbol, four of the subcarriers are dedicated to pilot signals in order to make the coherent
        detection robust against frequency offsets and phase noise. These pilot signals shall be put in subcarriers
        -21, -7, 7, and 21. The pilots shall be BPSK modulated by a pseudo-binary sequence to prevent the
        generation of spectral lines. The contribution of the pilot subcarriers to each OFDM symbol is described in
        17.3.5.10.


    l) Convert the subcarriers to time domain using inverse Fourier transform.
        Prepend to the Fourier-transformed waveform a circular extension of itself thus
        forming a GI, and truncate the resulting periodic waveform to a single OFDM symbol length by
        applying time domain windowing. Refer to 17.3.5.10 for details.

        Note: i am doing time domain windowing later, makes no sense to have it here

    Args:
        ofdm_symbol: 48 frequency domain values for each carrier
        index_in_package: determines pilot polarity, note that SIGNAL field has index of 0

    Returns:
        80 time domain samples (16 GI + 64 IFFT)
    """
    assert len(ofdm_symbol) == 48
    pilots = np.array([1, 1, 1, -1]) * PILOT_POLARITY[index_in_package % 127]
    carriers = [0] * 64
    carriers[-32] = 0
    carriers[-31] = 0
    carriers[-30] = 0
    carriers[-29] = 0
    carriers[-28] = 0
    carriers[-27] = 0
    carriers[-26] = ofdm_symbol[0]
    carriers[-25] = ofdm_symbol[1]
    carriers[-24] = ofdm_symbol[2]
    carriers[-23] = ofdm_symbol[3]
    carriers[-22] = ofdm_symbol[4]
    carriers[-21] = pilots[0]
    carriers[-20] = ofdm_symbol[5]
    carriers[-19] = ofdm_symbol[6]
    carriers[-18] = ofdm_symbol[7]
    carriers[-17] = ofdm_symbol[8]
    carriers[-16] = ofdm_symbol[9]
    carriers[-15] = ofdm_symbol[10]
    carriers[-14] = ofdm_symbol[11]
    carriers[-13] = ofdm_symbol[12]
    carriers[-12] = ofdm_symbol[13]
    carriers[-11] = ofdm_symbol[14]
    carriers[-10] = ofdm_symbol[15]
    carriers[-9] = ofdm_symbol[16]
    carriers[-8] = ofdm_symbol[17]
    carriers[-7] = pilots[1]
    carriers[-6] = ofdm_symbol[18]
    carriers[-5] = ofdm_symbol[19]
    carriers[-4] = ofdm_symbol[20]
    carriers[-3] = ofdm_symbol[21]
    carriers[-2] = ofdm_symbol[22]
    carriers[-1] = ofdm_symbol[23]
    carriers[0] = 0
    carriers[1] = ofdm_symbol[24]
    carriers[2] = ofdm_symbol[25]
    carriers[3] = ofdm_symbol[26]
    carriers[4] = ofdm_symbol[27]
    carriers[5] = ofdm_symbol[28]
    carriers[6] = ofdm_symbol[29]
    carriers[7] = pilots[2]
    carriers[8] = ofdm_symbol[30]
    carriers[9] = ofdm_symbol[31]
    carriers[10] = ofdm_symbol[32]
    carriers[11] = ofdm_symbol[33]
    carriers[12] = ofdm_symbol[34]
    carriers[13] = ofdm_symbol[35]
    carriers[14] = ofdm_symbol[36]
    carriers[15] = ofdm_symbol[37]
    carriers[16] = ofdm_symbol[38]
    carriers[17] = ofdm_symbol[39]
    carriers[18] = ofdm_symbol[40]
    carriers[19] = ofdm_symbol[41]
    carriers[20] = ofdm_symbol[42]
    carriers[21] = pilots[3]
    carriers[22] = ofdm_symbol[43]
    carriers[23] = ofdm_symbol[44]
    carriers[24] = ofdm_symbol[45]
    carriers[25] = ofdm_symbol[46]
    carriers[26] = ofdm_symbol[47]
    carriers[27] = 0
    carriers[28] = 0
    carriers[29] = 0
    carriers[30] = 0
    carriers[31] = 0

    ifft = np.fft.ifft(carriers)
    result = np.concatenate([ifft[-16:], ifft])  # add 16 samples of GI (guard interval)
    return result


l = deque(maxlen=16)
def demodulate_ofdm(samples: np.ndarray, equalizer:np.array, index_in_package: int) -> np.ndarray:
    """ Undo the 'modulate_ofdm'

    Args:
        samples: 80 time domain samples
        index_in_package: determines pilot polarity, note that SIGNAL field has index of 0

    Returns:
        OFDM symbol (48 frequency domain values) and 4 pilot symbols (frequency domain)

    """
    if index_in_package == 0:
        l.clear()

    assert len(samples) == 80
    carriers = np.fft.fft(samples[16:80]) * equalizer

    pilots = np.empty(4, dtype=complex)
    ofdm_symbol = np.empty(48, dtype=complex)
    ofdm_symbol[0] = carriers[-26]
    ofdm_symbol[1] = carriers[-25]
    ofdm_symbol[2] = carriers[-24]
    ofdm_symbol[3] = carriers[-23]
    ofdm_symbol[4] = carriers[-22]
    pilots[0] = carriers[-21]
    ofdm_symbol[5] = carriers[-20]
    ofdm_symbol[6] = carriers[-19]
    ofdm_symbol[7] = carriers[-18]
    ofdm_symbol[8] = carriers[-17]
    ofdm_symbol[9] = carriers[-16]
    ofdm_symbol[10] = carriers[-15]
    ofdm_symbol[11] = carriers[-14]
    ofdm_symbol[12] = carriers[-13]
    ofdm_symbol[13] = carriers[-12]
    ofdm_symbol[14] = carriers[-11]
    ofdm_symbol[15] = carriers[-10]
    ofdm_symbol[16] = carriers[-9]
    ofdm_symbol[17] = carriers[-8]
    pilots[1] = carriers[-7]
    ofdm_symbol[18] = carriers[-6]
    ofdm_symbol[19] = carriers[-5]
    ofdm_symbol[20] = carriers[-4]
    ofdm_symbol[21] = carriers[-3]
    ofdm_symbol[22] = carriers[-2]
    ofdm_symbol[23] = carriers[-1]
    ofdm_symbol[24] = carriers[1]
    ofdm_symbol[25] = carriers[2]
    ofdm_symbol[26] = carriers[3]
    ofdm_symbol[27] = carriers[4]
    ofdm_symbol[28] = carriers[5]
    ofdm_symbol[29] = carriers[6]
    pilots[2] = carriers[7]
    ofdm_symbol[30] = carriers[8]
    ofdm_symbol[31] = carriers[9]
    ofdm_symbol[32] = carriers[10]
    ofdm_symbol[33] = carriers[11]
    ofdm_symbol[34] = carriers[12]
    ofdm_symbol[35] = carriers[13]
    ofdm_symbol[36] = carriers[14]
    ofdm_symbol[37] = carriers[15]
    ofdm_symbol[38] = carriers[16]
    ofdm_symbol[39] = carriers[17]
    ofdm_symbol[40] = carriers[18]
    ofdm_symbol[41] = carriers[19]
    ofdm_symbol[42] = carriers[20]
    pilots[3] = carriers[21]
    ofdm_symbol[43] = carriers[22]
    ofdm_symbol[44] = carriers[23]
    ofdm_symbol[45] = carriers[24]
    ofdm_symbol[46] = carriers[25]
    ofdm_symbol[47] = carriers[26]

    # remove latent frequency offset by using pilot symbols
    pilots *= PILOT_POLARITY[index_in_package % 127]
    l.append(pilots[0])
    l.append(pilots[1])
    l.append(pilots[2])
    l.append(pilots[3])
    # print(l)
    mean_phase_offset = np.angle(np.mean(l))
    ofdm_symbol *= np.exp(-1j * mean_phase_offset)

    return ofdm_symbol


def test_ofdm_i18():
    # IEEE Std 802.11-2016 - Table I-19—Interleaved bits of first DATA symbol
    input = '0111011111110000111011111100010001110011000000001011111100010001000100001001101000011101000100100110111' \
            '00011100011110101011010010001101101101011100110000100001100000000000011011011001101101101 '
    from ieee80211phy.modulation import bits_to_symbols
    input_ofdm_symbol = bits_to_symbols(input, bits_per_symbol=4)
    output = modulate_ofdm(input_ofdm_symbol, index_in_package=1)

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

    # skipping first and last as they are involved in time windowing - which i will perform later
    np.testing.assert_equal(expected[1:-1], np.round(output[1:-1], 3))

    # test demodulation
    ofdm_symbol = demodulate_ofdm(output, equalizer=[1] * 64, index_in_package=1)
    np.testing.assert_allclose(ofdm_symbol, input_ofdm_symbol)
