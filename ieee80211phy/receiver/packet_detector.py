from scipy import signal
import numpy as np


def average(inputs, window_len):
    taps = [1 / window_len] * window_len
    return signal.lfilter(taps, [1.0], inputs)


def packet_detector(input):
    mult = input[:-16] * np.conjugate(input[16:])
    avg = average(mult, 32)
    autocorr = avg.real

    tmp = (input * np.conjugate(input)).real
    avg = average(tmp, 32)
    power = avg

    ratio = np.array(autocorr / power[:len(autocorr)])
    # ratio[ratio < 0.65] = 0
    detection = 0
    det = []
    for i in range(len(ratio)):
        if ratio[i] > 0.85:
            detection = 1
        elif ratio[i] < 0.65:
            detection = 0
        det.append(detection)

    return det
