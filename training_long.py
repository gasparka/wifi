import numpy as np


def get_long():
    long_train = [0, 0, 0, 0, 0, 0, 1, 1, -1, -1, 1, 1, -1, 1, -1, 1, 1, 1, 1, 1, 1, -1, -1, 1, 1, -1, 1, -1, 1, 1, 1, 1,
                  0,
                  1, -1, -1, 1, 1, -1, 1, -1, 1, -1, -1, -1, -1, -1, 1, 1, -1, -1, 1, -1, 1, -1, 1, 1, 1, 1, 0, 0, 0, 0, 0]
    long_time = []
    for i in range(64):
        v = 0.0
        for m in range(-32, 32):
            v += long_train[m + 32] * np.exp(1j * 2 * np.pi * i * m / 64)
        long_time.append(v / 64)

    full_long_time = np.array(long_time[32:] + long_time + long_time + long_time)[:161]
    full_long_time[0] *= 0.5
    full_long_time[160] *= 0.5
    return full_long_time.astype(np.complex64) # np.complex128 fails tests, because rounding gives different results


def test_():
    """ IEEE Std 802.11-2016 -  Table I-6—Time domain representation of the long sequence """
    result = np.round(get_long(), 3)
    expected = np.array([(-0.078 + 0j), (0.012 - 0.098j), (0.092 - 0.106j), (-0.092 - 0.115j), (-0.003 - 0.054j),
                (0.075 + 0.074j), (-0.127 + 0.021j), (-0.122 + 0.017j), (-0.035 + 0.151j), (-0.056 + 0.022j),
                (-0.06 - 0.081j), (0.07 - 0.014j), (0.082 - 0.092j), (-0.131 - 0.065j), (-0.057 - 0.039j),
                (0.037 - 0.098j), (0.062 + 0.062j), (0.119 + 0.004j), (-0.022 - 0.161j), (0.059 + 0.015j),
                (0.024 + 0.059j), (-0.137 + 0.047j), (0.001 + 0.115j), (0.053 - 0.004j), (0.098 + 0.026j),
                (-0.038 + 0.106j), (-0.115 + 0.055j), (0.06 + 0.088j), (0.021 - 0.028j), (0.097 - 0.083j),
                (0.04 + 0.111j), (-0.005 + 0.12j), (0.156 + 0j), (-0.005 - 0.12j), (0.04 - 0.111j), (0.097 + 0.083j),
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
                (0.06 + 0.088j), (0.021 - 0.028j), (0.097 - 0.083j), (0.04 + 0.111j), (-0.005 + 0.12j), (0.156 + 0j),
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
