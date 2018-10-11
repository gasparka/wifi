import numpy as np

"""
IEEE Std 802.11-2016 page 2283
Produce the PHY Preamble field, composed of 10 repetitions of a “short training sequence” (used
for AGC convergence, diversity selection, timing acquisition, and coarse frequency acquisition in
the receiver) and two repetitions of a “long training sequence” (used for channel estimation and fine
frequency acquisition in the receiver), preceded by a guard interval (GI). Refer to 17.3.3 for details.
"""


def short_training_sequence():
    """
    A short OFDM training symbol consists of 12 subcarriers.
    See '17.3.3 PHY preamble (SYNC)' IEEE Std 802.11-2016
    """
    pos = [0, 0, 0, 0,
           -1 - 1j, 0, 0, 0,
           -1 - 1j, 0, 0, 0,
           1 + 1j, 0, 0, 0,
           1 + 1j, 0, 0, 0,
           1 + 1j, 0, 0, 0,
           1 + 1j, 0, 0, 0,
           0, 0, 0, 0]

    neg = [0, 0, 0, 0,
           0, 0, 0, 0,
           1 + 1j, 0, 0, 0,
           -1 - 1j, 0, 0, 0,
           1 + 1j, 0, 0, 0,
           -1 - 1j, 0, 0, 0,
           -1 - 1j, 0, 0, 0,
           1 + 1j, 0, 0, 0, ]

    symbols = np.sqrt(13 / 6) * np.array(neg + pos)

    short_training = []
    for i in range(161):
        v = 0.0
        for m in range(-32, 32):
            v += symbols[m + 32] * np.exp(1j * 2 * np.pi * i * m / 64)
        short_training.append(v / 64)

    # time windowing method as discussed in "17.3.2.6 Discrete time implementation considerations"
    short_training[0] *= 0.5
    short_training[-1] *= 0.5
    return np.array(short_training).astype(np.complex64)


def long_training_sequence():
    """
    A long OFDM training symbol consists of 53 subcarriers (including the value 0 at dc).
    Two periods of the long sequence are transmitted for improved channel estimation accuracy.
    See '17.3.3 PHY preamble (SYNC)' IEEE Std 802.11-2016
    """
    long_train = [0, 0, 0, 0, 0, 0, 1, 1, -1, -1, 1, 1, -1, 1, -1, 1, 1, 1, 1, 1, 1, -1, -1, 1, 1, -1, 1, -1, 1, 1, 1,
                  1,
                  0,
                  1, -1, -1, 1, 1, -1, 1, -1, 1, -1, -1, -1, -1, -1, 1, 1, -1, -1, 1, -1, 1, -1, 1, 1, 1, 1, 0, 0, 0, 0,
                  0]
    long_time = []
    for i in range(64):
        v = 0.0
        for m in range(-32, 32):
            v += long_train[m + 32] * np.exp(1j * 2 * np.pi * i * m / 64)
        long_time.append(v / 64)

    full_long_time = (long_time[32:] + long_time + long_time + long_time)[:161]

    # time windowing method as discussed in "17.3.2.6 Discrete time implementation considerations"
    full_long_time[0] *= 0.5
    full_long_time[160] *= 0.5
    return np.array(full_long_time).astype(
        np.complex64)  # np.complex128 fails tests, because rounding gives different results


def preamble():
    """ Concatenates the short and long sequence """
    short = short_training_sequence()
    long = long_training_sequence()

    # time windowing method as discussed in "17.3.2.6 Discrete time implementation considerations"
    # connect the overlapping point
    short[-1] += long[0]
    long = long[1:]

    result = np.concatenate([short, long])
    return result


def test_short_training_sequence():
    """ IEEE Std 802.11-2016 - Table I-4—Time domain representation of the short sequence """
    result = np.round(short_training_sequence(), 3)
    expected = np.array(
        [(0.023 + 0.023j), (-0.132 + 0.002j), (-0.013 - 0.079j), (0.143 - 0.013j), (0.092 + 0j), (0.143 - 0.013j),
         (-0.013 - 0.079j), (-0.132 + 0.002j), (0.046 + 0.046j), (0.002 - 0.132j), (-0.079 - 0.013j), (-0.013 + 0.143j),
         0.092j, (-0.013 + 0.143j), (-0.079 - 0.013j), (0.002 - 0.132j), (0.046 + 0.046j), (-0.132 + 0.002j),
         (-0.013 - 0.079j), (0.143 - 0.013j), (0.092 + 0j), (0.143 - 0.013j), (-0.013 - 0.079j), (-0.132 + 0.002j),
         (0.046 + 0.046j), (0.002 - 0.132j), (-0.079 - 0.013j), (-0.013 + 0.143j), 0.092j, (-0.013 + 0.143j),
         (-0.079 - 0.013j), (0.002 - 0.132j), (0.046 + 0.046j), (-0.132 + 0.002j), (-0.013 - 0.079j), (0.143 - 0.013j),
         (0.092 + 0j), (0.143 - 0.013j), (-0.013 - 0.079j), (-0.132 + 0.002j), (0.046 + 0.046j), (0.002 - 0.132j),
         (-0.079 - 0.013j), (-0.013 + 0.143j), 0.092j, (-0.013 + 0.143j), (-0.079 - 0.013j), (0.002 - 0.132j),
         (0.046 + 0.046j), (-0.132 + 0.002j), (-0.013 - 0.079j), (0.143 - 0.013j), (0.092 + 0j), (0.143 - 0.013j),
         (-0.013 - 0.079j), (-0.132 + 0.002j), (0.046 + 0.046j), (0.002 - 0.132j), (-0.079 - 0.013j), (-0.013 + 0.143j),
         0.092j, (-0.013 + 0.143j), (-0.079 - 0.013j), (0.002 - 0.132j), (0.046 + 0.046j), (-0.132 + 0.002j),
         (-0.013 - 0.079j), (0.143 - 0.013j), (0.092 + 0j), (0.143 - 0.013j), (-0.013 - 0.079j), (-0.132 + 0.002j),
         (0.046 + 0.046j), (0.002 - 0.132j), (-0.079 - 0.013j), (-0.013 + 0.143j), 0.092j, (-0.013 + 0.143j),
         (-0.079 - 0.013j), (0.002 - 0.132j), (0.046 + 0.046j), (-0.132 + 0.002j), (-0.013 - 0.079j), (0.143 - 0.013j),
         (0.092 + 0j), (0.143 - 0.013j), (-0.013 - 0.079j), (-0.132 + 0.002j), (0.046 + 0.046j), (0.002 - 0.132j),
         (-0.079 - 0.013j), (-0.013 + 0.143j), 0.092j, (-0.013 + 0.143j), (-0.079 - 0.013j), (0.002 - 0.132j),
         (0.046 + 0.046j), (-0.132 + 0.002j), (-0.013 - 0.079j), (0.143 - 0.013j), (0.092 + 0j), (0.143 - 0.013j),
         (-0.013 - 0.079j), (-0.132 + 0.002j), (0.046 + 0.046j), (0.002 - 0.132j), (-0.079 - 0.013j), (-0.013 + 0.143j),
         0.092j, (-0.013 + 0.143j), (-0.079 - 0.013j), (0.002 - 0.132j), (0.046 + 0.046j), (-0.132 + 0.002j),
         (-0.013 - 0.079j), (0.143 - 0.013j), (0.092 + 0j), (0.143 - 0.013j), (-0.013 - 0.079j), (-0.132 + 0.002j),
         (0.046 + 0.046j), (0.002 - 0.132j), (-0.079 - 0.013j), (-0.013 + 0.143j), 0.092j, (-0.013 + 0.143j),
         (-0.079 - 0.013j), (0.002 - 0.132j), (0.046 + 0.046j), (-0.132 + 0.002j), (-0.013 - 0.079j), (0.143 - 0.013j),
         (0.092 + 0j), (0.143 - 0.013j), (-0.013 - 0.079j), (-0.132 + 0.002j), (0.046 + 0.046j), (0.002 - 0.132j),
         (-0.079 - 0.013j), (-0.013 + 0.143j), 0.092j, (-0.013 + 0.143j), (-0.079 - 0.013j), (0.002 - 0.132j),
         (0.046 + 0.046j), (-0.132 + 0.002j), (-0.013 - 0.079j), (0.143 - 0.013j), (0.092 + 0j), (0.143 - 0.013j),
         (-0.013 - 0.079j), (-0.132 + 0.002j), (0.046 + 0.046j), (0.002 - 0.132j), (-0.079 - 0.013j), (-0.013 + 0.143j),
         0.092j, (-0.013 + 0.143j), (-0.079 - 0.013j), (0.002 - 0.132j), (0.023 + 0.023j)], dtype=np.complex64)
    np.testing.assert_equal(expected, result)


def test_long_training_sequence():
    """ IEEE Std 802.11-2016 -  Table I-6—Time domain representation of the long sequence """
    result = np.round(long_training_sequence(), 3)
    expected = np.array([(-0.078 + 0j), (0.012 - 0.098j), (0.092 - 0.106j), (-0.092 - 0.115j), (-0.003 - 0.054j),
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
                         (0.097 - 0.083j), (0.04 + 0.111j), (-0.005 + 0.12j), (0.078 + 0j)], dtype=np.complex64)

    np.testing.assert_equal(expected, result)


def test_preamble():
    """
    Table I-22—Time domain representation of the short training sequence
    Table I-23—Time domain representation of the long training sequence
    """
    expect_short_train = np.array(
        [(0.023 + 0.023j), (-0.132 + 0.002j), (-0.013 - 0.079j), (0.143 - 0.013j), (0.092 + 0j),
         (0.143 - 0.013j), (-0.013 - 0.079j), (-0.132 + 0.002j), (0.046 + 0.046j), (0.002 - 0.132j),
         (-0.079 - 0.013j), (-0.013 + 0.143j), 0.092j, (-0.013 + 0.143j), (-0.079 - 0.013j),
         (0.002 - 0.132j), (0.046 + 0.046j), (-0.132 + 0.002j), (-0.013 - 0.079j), (0.143 - 0.013j),
         (0.092 + 0j), (0.143 - 0.013j), (-0.013 - 0.079j), (-0.132 + 0.002j), (0.046 + 0.046j),
         (0.002 - 0.132j), (-0.079 - 0.013j), (-0.013 + 0.143j), 0.092j, (-0.013 + 0.143j),
         (-0.079 - 0.013j), (0.002 - 0.132j), (0.046 + 0.046j), (-0.132 + 0.002j), (-0.013 - 0.079j),
         (0.143 - 0.013j), (0.092 + 0j), (0.143 - 0.013j), (-0.013 - 0.079j), (-0.132 + 0.002j),
         (0.046 + 0.046j), (0.002 - 0.132j), (-0.079 - 0.013j), (-0.013 + 0.143j), 0.092j,
         (-0.013 + 0.143j), (-0.079 - 0.013j), (0.002 - 0.132j), (0.046 + 0.046j), (-0.132 + 0.002j),
         (-0.013 - 0.079j), (0.143 - 0.013j), (0.092 + 0j), (0.143 - 0.013j), (-0.013 - 0.079j),
         (-0.132 + 0.002j), (0.046 + 0.046j), (0.002 - 0.132j), (-0.079 - 0.013j), (-0.013 + 0.143j),
         0.092j, (-0.013 + 0.143j), (-0.079 - 0.013j), (0.002 - 0.132j), (0.046 + 0.046j),
         (-0.132 + 0.002j), (-0.013 - 0.079j), (0.143 - 0.013j), (0.092 + 0j), (0.143 - 0.013j),
         (-0.013 - 0.079j), (-0.132 + 0.002j), (0.046 + 0.046j), (0.002 - 0.132j), (-0.079 - 0.013j),
         (-0.013 + 0.143j), 0.092j, (-0.013 + 0.143j), (-0.079 - 0.013j), (0.002 - 0.132j),
         (0.046 + 0.046j), (-0.132 + 0.002j), (-0.013 - 0.079j), (0.143 - 0.013j), (0.092 + 0j),
         (0.143 - 0.013j), (-0.013 - 0.079j), (-0.132 + 0.002j), (0.046 + 0.046j), (0.002 - 0.132j),
         (-0.079 - 0.013j), (-0.013 + 0.143j), 0.092j, (-0.013 + 0.143j), (-0.079 - 0.013j),
         (0.002 - 0.132j), (0.046 + 0.046j), (-0.132 + 0.002j), (-0.013 - 0.079j), (0.143 - 0.013j),
         (0.092 + 0j), (0.143 - 0.013j), (-0.013 - 0.079j), (-0.132 + 0.002j), (0.046 + 0.046j),
         (0.002 - 0.132j), (-0.079 - 0.013j), (-0.013 + 0.143j), 0.092j, (-0.013 + 0.143j),
         (-0.079 - 0.013j), (0.002 - 0.132j), (0.046 + 0.046j), (-0.132 + 0.002j), (-0.013 - 0.079j),
         (0.143 - 0.013j), (0.092 + 0j), (0.143 - 0.013j), (-0.013 - 0.079j), (-0.132 + 0.002j),
         (0.046 + 0.046j), (0.002 - 0.132j), (-0.079 - 0.013j), (-0.013 + 0.143j), 0.092j,
         (-0.013 + 0.143j), (-0.079 - 0.013j), (0.002 - 0.132j), (0.046 + 0.046j), (-0.132 + 0.002j),
         (-0.013 - 0.079j), (0.143 - 0.013j), (0.092 + 0j), (0.143 - 0.013j), (-0.013 - 0.079j),
         (-0.132 + 0.002j), (0.046 + 0.046j), (0.002 - 0.132j), (-0.079 - 0.013j), (-0.013 + 0.143j),
         0.092j, (-0.013 + 0.143j), (-0.079 - 0.013j), (0.002 - 0.132j), (0.046 + 0.046j),
         (-0.132 + 0.002j), (-0.013 - 0.079j), (0.143 - 0.013j), (0.092 + 0j), (0.143 - 0.013j),
         (-0.013 - 0.079j), (-0.132 + 0.002j), (0.046 + 0.046j), (0.002 - 0.132j), (-0.079 - 0.013j),
         (-0.013 + 0.143j), 0.092j, (-0.013 + 0.143j), (-0.079 - 0.013j), (0.002 - 0.132j)], dtype=np.complex64)

    expect_long_train = np.array(
        [(-0.055 + 0.023j), (0.012 - 0.098j), (0.092 - 0.106j), (-0.092 - 0.115j), (-0.003 - 0.054j),
         (0.075 + 0.074j), (-0.127 + 0.021j), (-0.122 + 0.017j), (-0.035 + 0.151j), (-0.056 + 0.022j),
         (-0.06 - 0.081j), (0.07 - 0.014j), (0.082 - 0.092j), (-0.131 - 0.065j), (-0.057 - 0.039j),
         (0.037 - 0.098j), (0.062 + 0.062j), (0.119 + 0.004j), (-0.022 - 0.161j), (0.059 + 0.015j),
         (0.024 + 0.059j), (-0.137 + 0.047j), (0.001 + 0.115j), (0.053 - 0.004j), (0.098 + 0.026j),
         (-0.038 + 0.106j), (-0.115 + 0.055j), (0.06 + 0.088j), (0.021 - 0.028j), (0.097 - 0.083j),
         (0.04 + 0.111j), (-0.005 + 0.12j), (0.156 + 0j), (-0.005 - 0.12j), (0.04 - 0.111j),
         (0.097 + 0.083j), (0.021 + 0.028j), (0.06 - 0.088j), (-0.115 - 0.055j), (-0.038 - 0.106j),
         (0.098 - 0.026j), (0.053 + 0.004j), (0.001 - 0.115j), (-0.137 - 0.047j), (0.024 - 0.059j),
         (0.059 - 0.015j), (-0.022 + 0.161j), (0.119 - 0.004j), (0.062 - 0.062j), (0.037 + 0.098j),
         (-0.057 + 0.039j), (-0.131 + 0.065j), (0.082 + 0.092j), (0.07 + 0.014j), (-0.06 + 0.081j),
         (-0.056 - 0.022j), (-0.035 - 0.151j), (-0.122 - 0.017j), (-0.127 - 0.021j), (0.075 - 0.074j),
         (-0.003 + 0.054j), (-0.092 + 0.115j), (0.092 + 0.106j), (0.012 + 0.098j), (-0.156 + 0j),
         (0.012 - 0.098j), (0.092 - 0.106j), (-0.092 - 0.115j), (-0.003 - 0.054j), (0.075 + 0.074j),
         (-0.127 + 0.021j), (-0.122 + 0.017j), (-0.035 + 0.151j), (-0.056 + 0.022j), (-0.06 - 0.081j),
         (0.07 - 0.014j), (0.082 - 0.092j), (-0.131 - 0.065j), (-0.057 - 0.039j), (0.037 - 0.098j),
         (0.062 + 0.062j), (0.119 + 0.004j), (-0.022 - 0.161j), (0.059 + 0.015j), (0.024 + 0.059j),
         (-0.137 + 0.047j), (0.001 + 0.115j), (0.053 - 0.004j), (0.098 + 0.026j), (-0.038 + 0.106j),
         (-0.115 + 0.055j), (0.06 + 0.088j), (0.021 - 0.028j), (0.097 - 0.083j), (0.04 + 0.111j),
         (-0.005 + 0.12j), (0.156 + 0j), (-0.005 - 0.12j), (0.04 - 0.111j), (0.097 + 0.083j),
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
         (0.06 + 0.088j), (0.021 - 0.028j), (0.097 - 0.083j), (0.04 + 0.111j), (-0.005 + 0.12j)], dtype=np.complex64)

    expect = np.concatenate([expect_short_train, expect_long_train])
    output = np.round(preamble(), 3)

    np.testing.assert_equal(expect, output[
                                    :-1])  # throwing away last sample (:-1) because this is meant to be time windowed with the signal field
