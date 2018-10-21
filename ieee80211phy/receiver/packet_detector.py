from ieee80211phy.transmitter.main import tx_generator
from scipy import signal
import numpy as np


def moving_average(inputs, window_len):
    taps = [1 / window_len] * window_len
    return signal.lfilter(taps, [1.0], inputs)


def packet_detector(input, debug=False):
    autocorr = (input[:-16] * np.conjugate(input[16:])).real
    autocorr = moving_average(autocorr, 128)

    power = (input * np.conjugate(input)).real
    power = moving_average(power, 128)

    # valid packet will have ratio near 1.0
    ratio = np.array(autocorr / power[:len(autocorr)])
    detection = 0
    det = []
    threshold = 0.8
    for i in range(len(ratio)):
        if ratio[i] > 0.8:
            detection = 1
        elif ratio[i] < 0.8:
            detection = 0
        det.append(detection)

    # return the first packet
    start_of_long_training_index = 0
    try:
        start_of_long_training_index = np.where(np.diff(det) == -1)[0][0] - 10 # -10 to correct the position!
    except:
        pass

    if debug:
        import matplotlib.pyplot as plt
        plt.figure(figsize=(9.75, 5))
        plt.plot(autocorr, label='auto-correlation')
        plt.plot(power, label='power')
        plt.legend()
        plt.tight_layout()
        plt.grid()

        plt.figure(figsize=(9.75, 5))
        plt.plot(ratio, label='autocorr / power')
        plt.plot(det, label='detection')
        plt.stem([start_of_long_training_index], [threshold])
        plt.legend()
        plt.tight_layout()
        plt.grid()
    return start_of_long_training_index


def test_():
    np.random.seed(0)
    data = ''.join('1' if x else '0' for x in np.random.randint(2, size=20906))
    tx, maps, ofdm_syms = tx_generator(data, data_rate=36)

    index = packet_detector(tx)
    expect = 160
    error = expect - index
    assert -1 <= error <= 1