import logging
import os
from dataclasses import dataclass
from typing import Tuple

import numpy as np
import pytest

from wifi import bits
from wifi.util import moving_average, awgn, evm_db2

logger = logging.getLogger(__name__)
Trace = {}


def packet_detector(input):
    """
    Returns index to start of first long training symbol
    """
    autocorr = (input[:-16] * np.conjugate(input[16:])).real
    autocorr = moving_average(autocorr, 8)
    Trace['detector_autocorr'] = autocorr

    power = (input * np.conjugate(input)).real
    power = moving_average(power, 8)
    Trace['detector_power'] = power

    ratio = [1.0 if x else -1.0 for x in autocorr >= 0.5 * power[:len(autocorr)]]
    ratio = moving_average(ratio, 128)  # valid packet will have ratio 1.0
    Trace['detector_ratio'] = ratio

    # extract sampling points
    state = 0
    timings = []
    for i in range(len(ratio)):
        if ratio[i] >= 0.95:
            state = 1
        elif ratio[i] < 0.77:
            if state == 1:
                timings.append(i + 31)
            state = 0

    logger.info(f'Found packets at: {timings}')
    return timings


@dataclass
class Packet:
    equalizer: np.ndarray
    signal_field_symbols: Tuple[np.ndarray, int]
    data_rate: int
    n_ofdm_symbols: int
    length_bytes: int
    modulation: str
    coding_rate: str
    data_symbols: Tuple[np.ndarray, int]
    bits: str


def test_packet_detector():
    iq = np.load(dir_path + '../data/limemini_lime_air.npy')
    iq = np.hstack([iq, iq, iq, iq])
    indexes = packet_detector(iq)
    assert indexes == [48075, 148075, 248075, 348075]


@pytest.mark.parametrize('data_rate', [6, 9, 12, 18, 24, 36, 48, 54])
def test_loopback(data_rate):
    np.random.seed(0)
    data_bits = bits(
        '0x0402002E006008CD37A60020D6013CF1006008AD3BAF00004A6F792C2062726967687420737061726B206F6620646976696E6974792C0A4461756768746572206F6620456C797369756D2C0A466972652D696E73697265642077652074726561673321B6')
    iq = transmit(data_bits, data_rate)
    iq = awgn(iq, 20)

    i = packet_detector(iq)[0]
    packet = receive(iq[i - 2:])
    assert packet.bits == data_bits


dir_path = os.path.dirname(os.path.realpath(__file__)) + '/'


def test_evm_limemini():
    """
    Recorded with a single LimeSDR-Mini - same device guarantees 0 freq and sampling offset!
    Package 4095 bytes, 228 OFDM symbols, data_rate=36, modulation=16-QAM, coding_rate=3/4
    """

    iq = np.load(dir_path + '../data/limemini_air.npy')
    i = packet_detector(iq)[0]
    packet = receive(iq[i - 2:])

    evm = evm_db2(*packet.data_symbols)
    assert int(evm) == -23


def test_evm2():
    iq = np.load(dir_path + '../data/sym8_rate24.npy')
    i = packet_detector(iq)[0]
    packet = receive(iq[i - 2:])

    evm = evm_db2(*packet.data_symbols)
    assert int(evm) == -19


def test_evm3():
    """ This one has weird Sin shape EVM vs time.. sampling errro? """
    iq = np.load(dir_path + '../data/sym130_rate24.npy')
    i = packet_detector(iq)[0]
    packet = receive(iq[i - 2:])

    evm = evm_db2(*packet.data_symbols)
    assert int(evm) == -18


def test_evm4():
    """ This one has weird Sin shape EVM vs time.. sampling errro? """
    iq = np.load(dir_path + '../data/sym173_rate18.npy')
    i = packet_detector(iq)[0]
    packet = receive(iq[i - 3:])

    evm = evm_db2(*packet.data_symbols)
    assert int(evm) == -18


def test_evm5():
    iq = np.load(dir_path + '../data/sym16_rate48.npy')
    i = packet_detector(iq)[0]
    packet = receive(iq[i - 3:])

    evm = evm_db2(*packet.data_symbols)
    assert int(evm) == -19

# def test_limemini_lime_air():
#     """
#     Lime-Mini as TX and Lime as RX. First test with different devices, resulted in frequency offset - that
#     was best corrected using the pilot symbols!
#     """
#
#     iq = np.load('/home/gaspar/git/wifi/data/limemini_lime_air.npy')
#     r = Receiver(sample_advance=-3)
#     symbols = r.main(iq, n_symbols=229)
#
#     from wifi.transmitter.subcarrier_modulation_mapping import mapper_decide
#     reference_symbols = np.array([[mapper_decide(j, 4) for j in x] for x in symbols])
#     evm = evm_db(symbols, reference_symbols)
#     assert int(evm) == -26
