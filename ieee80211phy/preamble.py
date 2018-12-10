import numpy as np


def short_training_symbol() -> np.ndarray:
    """
    Short training symbol is used for AGC convergence, diversity selection, timing acquisition, and coarse frequency acquisition in
    the receiver. Refer to 17.3.3 for details.

    Returns:
         short training symbol in frequency domain
    """
    carriers = [0 + 0j] * 64
    carriers[-32] = 0
    carriers[-31] = 0
    carriers[-30] = 0
    carriers[-29] = 0
    carriers[-28] = 0
    carriers[-27] = 0
    carriers[-26] = 0
    carriers[-25] = 0
    carriers[-24] = 1 + 1j
    carriers[-23] = 0
    carriers[-22] = 0
    carriers[-21] = 0
    carriers[-20] = -1 - 1j
    carriers[-19] = 0
    carriers[-18] = 0
    carriers[-17] = 0
    carriers[-16] = 1 + 1j
    carriers[-15] = 0
    carriers[-14] = 0
    carriers[-13] = 0
    carriers[-12] = -1 - 1j
    carriers[-11] = 0
    carriers[-10] = 0
    carriers[-9] = 0
    carriers[-8] = -1 - 1j
    carriers[-7] = 0
    carriers[-6] = 0
    carriers[-5] = 0
    carriers[-4] = 1 + 1j
    carriers[-3] = 0
    carriers[-2] = 0
    carriers[-1] = 0
    carriers[0] = 0
    carriers[1] = 0
    carriers[2] = 0
    carriers[3] = 0
    carriers[4] = -1 - 1j
    carriers[5] = 0
    carriers[6] = 0
    carriers[7] = 0
    carriers[8] = -1 - 1j
    carriers[9] = 0
    carriers[10] = 0
    carriers[11] = 0
    carriers[12] = 1 + 1j
    carriers[13] = 0
    carriers[14] = 0
    carriers[15] = 0
    carriers[16] = 1 + 1j
    carriers[17] = 0
    carriers[18] = 0
    carriers[19] = 0
    carriers[20] = 1 + 1j
    carriers[21] = 0
    carriers[22] = 0
    carriers[23] = 0
    carriers[24] = 1 + 1j
    carriers[25] = 0
    carriers[26] = 0
    carriers[27] = 0
    carriers[28] = 0
    carriers[29] = 0
    carriers[30] = 0
    carriers[31] = 0
    return np.array(carriers) * np.sqrt(13 / 6)


def short_training_sequence() -> np.ndarray:
    """

    Returns:
        160 time domain samples of long training sequence
    """

    symbol = np.fft.ifft(short_training_symbol())
    full_long_time = np.concatenate([symbol[32:], symbol, symbol])  # two symbols plus 32 samples of GI
    return full_long_time


def long_training_symbol() -> np.ndarray:
    """ This is sent in a preamble as a known symbol, which is used for channel estimation i.e.
    receiver can compare how the channel has distorted these symbols and figure out the counter actions.

    Returns:
        long training symbol in frequency domain
    """
    carriers = [0 + 0j] * 64
    carriers[-32] = 0
    carriers[-31] = 0
    carriers[-30] = 0
    carriers[-29] = 0
    carriers[-28] = 0
    carriers[-27] = 0
    carriers[-26] = 1
    carriers[-25] = 1
    carriers[-24] = -1
    carriers[-23] = -1
    carriers[-22] = 1
    carriers[-21] = 1
    carriers[-20] = -1
    carriers[-19] = 1
    carriers[-18] = -1
    carriers[-17] = 1
    carriers[-16] = 1
    carriers[-15] = 1
    carriers[-14] = 1
    carriers[-13] = 1
    carriers[-12] = 1
    carriers[-11] = -1
    carriers[-10] = -1
    carriers[-9] = 1
    carriers[-8] = 1
    carriers[-7] = -1
    carriers[-6] = 1
    carriers[-5] = -1
    carriers[-4] = 1
    carriers[-3] = 1
    carriers[-2] = 1
    carriers[-1] = 1
    carriers[0] = 0
    carriers[1] = 1
    carriers[2] = -1
    carriers[3] = -1
    carriers[4] = 1
    carriers[5] = 1
    carriers[6] = -1
    carriers[7] = 1
    carriers[8] = -1
    carriers[9] = 1
    carriers[10] = -1
    carriers[11] = -1
    carriers[12] = -1
    carriers[13] = -1
    carriers[14] = -1
    carriers[15] = 1
    carriers[16] = 1
    carriers[17] = -1
    carriers[18] = -1
    carriers[19] = 1
    carriers[20] = -1
    carriers[21] = 1
    carriers[22] = -1
    carriers[23] = 1
    carriers[24] = 1
    carriers[25] = 1
    carriers[26] = 1
    carriers[27] = 0
    carriers[28] = 0
    carriers[29] = 0
    carriers[30] = 0
    carriers[31] = 0
    return np.array(carriers)


def long_training_sequence():
    """
    Two periods of the long sequence are transmitted for improved channel estimation accuracy.
    See '17.3.3 PHY preamble (SYNC)' IEEE Std 802.11-2016
    """

    symbol = np.fft.ifft(long_training_symbol())
    full_long_time = np.concatenate([symbol[32:], symbol, symbol])  # two symbols plus 32 samples of GI
    full_l = np.concatenate([symbol[48:], symbol, symbol[48:], symbol])
    return full_long_time


def test_short_training_sequence():
    """ IEEE Std 802.11-2016 - Table I-4—Time domain representation of the short sequence """
    result = np.round(short_training_sequence(), 3)
    expected = np.array(
        [(0.023 + 0.023j), (-0.132 + 0.002j), (-0.013 - 0.079j), (0.143 - 0.013j), (0.092 + 0j),
         (0.143 - 0.013j),
         (-0.013 - 0.079j), (-0.132 + 0.002j), (0.046 + 0.046j), (0.002 - 0.132j), (-0.079 - 0.013j),
         (-0.013 + 0.143j),
         0.092j, (-0.013 + 0.143j), (-0.079 - 0.013j), (0.002 - 0.132j), (0.046 + 0.046j),
         (-0.132 + 0.002j),
         (-0.013 - 0.079j), (0.143 - 0.013j), (0.092 + 0j), (0.143 - 0.013j), (-0.013 - 0.079j),
         (-0.132 + 0.002j),
         (0.046 + 0.046j), (0.002 - 0.132j), (-0.079 - 0.013j), (-0.013 + 0.143j), 0.092j,
         (-0.013 + 0.143j),
         (-0.079 - 0.013j), (0.002 - 0.132j), (0.046 + 0.046j), (-0.132 + 0.002j), (-0.013 - 0.079j),
         (0.143 - 0.013j),
         (0.092 + 0j), (0.143 - 0.013j), (-0.013 - 0.079j), (-0.132 + 0.002j), (0.046 + 0.046j),
         (0.002 - 0.132j),
         (-0.079 - 0.013j), (-0.013 + 0.143j), 0.092j, (-0.013 + 0.143j), (-0.079 - 0.013j),
         (0.002 - 0.132j),
         (0.046 + 0.046j), (-0.132 + 0.002j), (-0.013 - 0.079j), (0.143 - 0.013j), (0.092 + 0j),
         (0.143 - 0.013j),
         (-0.013 - 0.079j), (-0.132 + 0.002j), (0.046 + 0.046j), (0.002 - 0.132j), (-0.079 - 0.013j),
         (-0.013 + 0.143j),
         0.092j, (-0.013 + 0.143j), (-0.079 - 0.013j), (0.002 - 0.132j), (0.046 + 0.046j),
         (-0.132 + 0.002j),
         (-0.013 - 0.079j), (0.143 - 0.013j), (0.092 + 0j), (0.143 - 0.013j), (-0.013 - 0.079j),
         (-0.132 + 0.002j),
         (0.046 + 0.046j), (0.002 - 0.132j), (-0.079 - 0.013j), (-0.013 + 0.143j), 0.092j,
         (-0.013 + 0.143j),
         (-0.079 - 0.013j), (0.002 - 0.132j), (0.046 + 0.046j), (-0.132 + 0.002j), (-0.013 - 0.079j),
         (0.143 - 0.013j),
         (0.092 + 0j), (0.143 - 0.013j), (-0.013 - 0.079j), (-0.132 + 0.002j), (0.046 + 0.046j),
         (0.002 - 0.132j),
         (-0.079 - 0.013j), (-0.013 + 0.143j), 0.092j, (-0.013 + 0.143j), (-0.079 - 0.013j),
         (0.002 - 0.132j),
         (0.046 + 0.046j), (-0.132 + 0.002j), (-0.013 - 0.079j), (0.143 - 0.013j), (0.092 + 0j),
         (0.143 - 0.013j),
         (-0.013 - 0.079j), (-0.132 + 0.002j), (0.046 + 0.046j), (0.002 - 0.132j), (-0.079 - 0.013j),
         (-0.013 + 0.143j),
         0.092j, (-0.013 + 0.143j), (-0.079 - 0.013j), (0.002 - 0.132j), (0.046 + 0.046j),
         (-0.132 + 0.002j),
         (-0.013 - 0.079j), (0.143 - 0.013j), (0.092 + 0j), (0.143 - 0.013j), (-0.013 - 0.079j),
         (-0.132 + 0.002j),
         (0.046 + 0.046j), (0.002 - 0.132j), (-0.079 - 0.013j), (-0.013 + 0.143j), 0.092j,
         (-0.013 + 0.143j),
         (-0.079 - 0.013j), (0.002 - 0.132j), (0.046 + 0.046j), (-0.132 + 0.002j), (-0.013 - 0.079j),
         (0.143 - 0.013j),
         (0.092 + 0j), (0.143 - 0.013j), (-0.013 - 0.079j), (-0.132 + 0.002j), (0.046 + 0.046j),
         (0.002 - 0.132j),
         (-0.079 - 0.013j), (-0.013 + 0.143j), 0.092j, (-0.013 + 0.143j), (-0.079 - 0.013j),
         (0.002 - 0.132j),
         (0.046 + 0.046j), (-0.132 + 0.002j), (-0.013 - 0.079j), (0.143 - 0.013j), (0.092 + 0j),
         (0.143 - 0.013j),
         (-0.013 - 0.079j), (-0.132 + 0.002j), (0.046 + 0.046j), (0.002 - 0.132j), (-0.079 - 0.013j),
         (-0.013 + 0.143j),

         0.092j, (-0.013 + 0.143j), (-0.079 - 0.013j), (0.002 - 0.132j), (0.023 + 0.023j)])

    # About skipping some samples: IEE tests vectors include time-windowing - i am doing this step later
    np.testing.assert_equal(expected[1:-2], result[1:-1])


def test_long_training_sequence():
    """ IEEE Std 802.11-2016 -  Table I-6—Time domain representation of the long sequence """
    result = np.round(long_training_sequence(), 3)
    expected = np.array(
        [(-0.078 + 0j), (0.012 - 0.098j), (0.092 - 0.106j), (-0.092 - 0.115j), (-0.003 - 0.054j),
         (0.075 + 0.074j), (-0.127 + 0.021j), (-0.122 + 0.017j), (-0.035 + 0.151j), (-0.056 + 0.022j),
         (-0.06 - 0.081j), (0.07 - 0.014j), (0.082 - 0.092j), (-0.131 - 0.065j), (-0.057 - 0.039j),
         (0.037 - 0.098j), (0.062 + 0.062j), (0.119 + 0.004j), (-0.022 - 0.161j), (0.059 + 0.015j),
         (0.024 + 0.059j), (-0.137 + 0.047j), (0.001 + 0.115j), (0.053 - 0.004j), (0.098 + 0.026j),
         (-0.038 + 0.106j), (-0.115 + 0.055j), (0.06 + 0.088j), (0.021 - 0.028j), (0.097 - 0.083j),
         (0.04 + 0.111j), (-0.005 + 0.12j), (0.156 + 0j), (-0.005 - 0.12j), (0.04 - 0.111j),
         (0.097 + 0.083j),
         (0.021 + 0.028j), (0.06 - 0.088j), (-0.115 - 0.055j), (-0.038 - 0.106j), (0.098 - 0.026j),
         (0.053 + 0.004j), (0.001 - 0.115j), (-0.137 - 0.047j), (0.024 - 0.059j), (0.059 - 0.015j),
         (-0.022 + 0.161j), (0.119 - 0.004j), (0.062 - 0.062j), (0.037 + 0.098j), (-0.057 + 0.039j),
         (-0.131 + 0.065j), (0.082 + 0.092j), (0.07 + 0.014j), (-0.06 + 0.081j), (-0.056 - 0.022j),
         (-0.035 - 0.151j), (-0.122 - 0.017j), (-0.127 - 0.021j), (0.075 - 0.074j), (-0.003 + 0.054j),
         (-0.092 + 0.115j), (0.092 + 0.106j), (0.012 + 0.098j), (-0.156 + 0j), (0.012 - 0.098j),
         (0.092 - 0.106j), (-0.092 - 0.115j), (-0.003 - 0.054j), (0.075 + 0.074j), (-0.127 + 0.021j),
         (-0.122 + 0.017j), (-0.035 + 0.151j), (-0.056 + 0.022j), (-0.06 - 0.081j), (0.07 - 0.014j),
         (0.082 - 0.092j), (-0.131 - 0.065j), (-0.057 - 0.039j), (0.037 - 0.098j), (0.062 + 0.062j),
         (0.119 + 0.004j), (-0.022 - 0.161j), (0.059 + 0.015j), (0.024 + 0.059j), (-0.137 + 0.047j),
         (0.001 + 0.115j), (0.053 - 0.004j), (0.098 + 0.026j), (-0.038 + 0.106j), (-0.115 + 0.055j),
         (0.06 + 0.088j), (0.021 - 0.028j), (0.097 - 0.083j), (0.04 + 0.111j), (-0.005 + 0.12j),
         (0.156 + 0j),
         (-0.005 - 0.12j), (0.04 - 0.111j), (0.097 + 0.083j), (0.021 + 0.028j), (0.06 - 0.088j),
         (-0.115 - 0.055j), (-0.038 - 0.106j), (0.098 - 0.026j), (0.053 + 0.004j), (0.001 - 0.115j),
         (-0.137 - 0.047j), (0.024 - 0.059j), (0.059 - 0.015j), (-0.022 + 0.161j), (0.119 - 0.004j),
         (0.062 - 0.062j), (0.037 + 0.098j), (-0.057 + 0.039j), (-0.131 + 0.065j), (0.082 + 0.092j),
         (0.07 + 0.014j), (-0.06 + 0.081j), (-0.056 - 0.022j), (-0.035 - 0.151j), (-0.122 - 0.017j),
         (-0.127 - 0.021j), (0.075 - 0.074j), (-0.003 + 0.054j), (-0.092 + 0.115j), (0.092 + 0.106j),
         (0.012 + 0.098j), (-0.156 + 0j), (0.012 - 0.098j), (0.092 - 0.106j), (-0.092 - 0.115j),
         (-0.003 - 0.054j), (0.075 + 0.074j), (-0.127 + 0.021j), (-0.122 + 0.017j), (-0.035 + 0.151j),
         (-0.056 + 0.022j), (-0.06 - 0.081j), (0.07 - 0.014j), (0.082 - 0.092j), (-0.131 - 0.065j),
         (-0.057 - 0.039j), (0.037 - 0.098j), (0.062 + 0.062j), (0.119 + 0.004j), (-0.022 - 0.161j),
         (0.059 + 0.015j), (0.024 + 0.059j), (-0.137 + 0.047j), (0.001 + 0.115j), (0.053 - 0.004j),
         (0.098 + 0.026j), (-0.038 + 0.106j), (-0.115 + 0.055j), (0.06 + 0.088j), (0.021 - 0.028j),
         (0.097 - 0.083j), (0.04 + 0.111j), (-0.005 + 0.12j), (0.078 + 0j)])

    # About skipping some samples: IEE tests vectors include time-windowing - i am doing this step later
    np.testing.assert_equal(expected[1:-2], result[1:-1])
