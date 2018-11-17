import logging
import numpy as np

from ieee80211phy.preamble import long_training_symbol
from ieee80211phy.util import moving_average

logger = logging.getLogger(__name__)


def packet_detector(input):
    """
    Returns index to start of first long training symbol
    """
    autocorr = (input[:-16] * np.conjugate(input[16:])).real
    autocorr = moving_average(autocorr, 8)

    power = (input * np.conjugate(input)).real
    power = moving_average(power, 8)

    ratio = [1.0 if x else -1.0 for x in autocorr >= 0.5 * power[:len(autocorr)]]
    ratio = moving_average(ratio, 128)  # valid packet will have ratio 1.0

    # extract sampling points
    state = 0
    timings = []
    for i in range(len(ratio)):
        if ratio[i] >= 1.0:
            state = 1
        elif ratio[i] < 0.77:
            if state == 1:
                timings.append(i + 32)
            state = 0

    logger.info(f'Found packets at: {timings}')
    return timings


def receiver(iq, i):
    # average of two training symbols
    avg_train = (iq[i:i + 64] + iq[i + 64:i + 128]) / 2
    channel_estimate = np.fft.fft(avg_train) / long_training_symbol()
    equalizer = 1 / channel_estimate

    # decode the signal field

    # parse the payload
    data_rx = iq[start_of_long_training + 160:start_of_long_training + 160 + (n_symbols * 80)]

    # parse the signal field
    signal_field = data_rx[:80]
    start = 16 + self.sample_advance
    symbols = np.fft.fft(signal_field[start:start + 64])
    equalized_symbols = symbols * self.equalizer_taps


def test_packet_detector():
    iq = np.load('/home/gaspar/git/ieee80211phy/data/limemini_wire_loopback.npy')
    iq = np.hstack([iq, iq, iq, iq])
    indexes = packet_detector(iq)
    assert indexes == [95027, 295027, 495027, 695027]