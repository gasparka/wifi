import numpy as np
from wifi.preambler import long_training_symbol


def undo(long_training, carriers):
    long_training = np.array(long_training[32:])
    avg_training = (long_training[:64] + long_training[64:128]) / 2
    channel_estimate = np.fft.fft(avg_training) / long_training_symbol()
    equalizer = 1 / channel_estimate

    result = np.array(carriers) * equalizer
    return result.tolist()
