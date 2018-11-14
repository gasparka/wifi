import logging
from ieee80211phy.transmitter.main import tx_generator
import numpy as np

from ieee80211phy.util import moving_average

logger = logging.getLogger(__name__)


def packet_detector(input):
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
                timings.append(i)
            state = 0

    logger.info(f'Found packets at: {timings}')
    return timings


def test_():
    np.random.seed(0)
    data = ''.join('1' if x else '0' for x in np.random.randint(2, size=20906))
    tx, maps, ofdm_syms = tx_generator(data, data_rate=36)

    index = packet_detector(tx)
    assert index == [161]


def test_multi():
    iq = np.load('/home/gaspar/git/ieee80211phy/data/limemini_wire_loopback.npy')
    iq = np.hstack([iq, iq, iq, iq])
    indexes = packet_detector(iq)
    assert indexes == [95027, 295027, 495027, 695027]
