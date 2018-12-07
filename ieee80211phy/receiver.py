import logging
from dataclasses import dataclass
from typing import Tuple

import numpy as np
import pytest

from ieee80211phy import scrambler
from ieee80211phy.conv_coding import conv_decode
from ieee80211phy.interleaving import interleave
from ieee80211phy.modulation import symbols_to_bits
from ieee80211phy.ofdm import demodulate_ofdm_factory
from ieee80211phy.preamble import long_training_symbol
from ieee80211phy.scrambler import scrambler
from ieee80211phy.signal_field import decode_signal_field
from ieee80211phy.transmitter import get_params_from_rate, transmitter
from ieee80211phy.util import moving_average, hex_to_bitstr, awgn, evm_db2

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


def receiver(iq):
    """ Channel estimation - calculate how much the known symbols have changed and produce inverse channel """
    avg_train = (iq[:64] + iq[64:128]) / 2
    channel_estimate = np.fft.fft(avg_train) / long_training_symbol()
    equalizer = 1 / channel_estimate

    """ Signal field demodulation - this gives us the data rate for the payload and also the payload length """
    ofdm_demodulator = demodulate_ofdm_factory()
    signal = iq[128: 128 + 80]
    signal_symbols = ofdm_demodulator(signal, equalizer, index_in_package=0)
    bits = symbols_to_bits(signal_symbols, bits_per_symbol=1)
    bits = interleave(bits, coded_bits_symbol=48, coded_bits_subcarrier=1, undo=True)
    bits = conv_decode(bits)
    data_rate, length_bytes = decode_signal_field(bits)
    modulation, coding_rate, coded_bits_subcarrier, coded_bits_symbol, data_bits_symbol = get_params_from_rate(
        data_rate)
    n_ofdm_symbols = int(np.ceil((length_bytes * 8 + 22) / data_bits_symbol))
    logger.info(
        f'Package {length_bytes} bytes -> {n_ofdm_symbols} OFDM symbols @ {data_rate}MB/s ({modulation}, {coding_rate})')

    """ Payload demodulation """
    signal_end = 128 + 80
    data_groups = iq[signal_end: signal_end + (80 * n_ofdm_symbols)].reshape((-1, 80))
    data_symbols = np.array([ofdm_demodulator(group, equalizer, index_in_package=1 + i)
                             for i, group in enumerate(data_groups)])

    """ Symbols to bits flow """
    bits = [symbols_to_bits(symbol, bits_per_symbol=coded_bits_subcarrier)
            for symbol in data_symbols]
    bits = ''.join(interleave(b, coded_bits_symbol, coded_bits_subcarrier, undo=True) for b in bits)
    bits = conv_decode(bits, coding_rate)
    bits = scrambler(bits)
    bits = bits[16:16 + length_bytes * 8]
    # data_symbols = None

    result = Packet(equalizer,
                    (signal_symbols, 1),
                    data_rate,
                    n_ofdm_symbols,
                    length_bytes,
                    modulation,
                    coding_rate,
                    (data_symbols, coded_bits_subcarrier),
                    bits)
    return result


def test_packet_detector():
    iq = np.load('/home/gaspar/git/ieee80211phy/data/limemini_lime_air.npy')
    iq = np.hstack([iq, iq, iq, iq])
    indexes = packet_detector(iq)
    assert indexes == [48075, 148075, 248075, 348075]


@pytest.mark.parametrize('data_rate', [6, 9, 12, 18, 24, 36, 48, 54])
def test_loopback(data_rate):
    np.random.seed(0)
    tx = '0x0402002E006008CD37A60020D6013CF1006008AD3BAF00004A6F792C2062726967687420737061726B206F6620646976696E6974792C0A4461756768746572206F6620456C797369756D2C0A466972652D696E73697265642077652074726561673321B6'
    tx_bits = hex_to_bitstr(tx)
    iq = transmitter(tx_bits, data_rate)
    iq = awgn(iq, 20)

    i = packet_detector(iq)[0]
    packet = receiver(iq[i - 2:])
    assert packet.bits == tx_bits


def test_evm_limemini():
    """
    Recorded with a single LimeSDR-Mini - same device guarantees 0 freq and sampling offset!
    Package 4095 bytes, 228 OFDM symbols, data_rate=36, modulation=16-QAM, coding_rate=3/4
    """

    iq = np.load('/home/gaspar/git/ieee80211phy/data/limemini_air.npy')
    i = packet_detector(iq)[0]
    packet = receiver(iq[i - 2:])

    evm = evm_db2(*packet.data_symbols)
    assert int(evm) == -23


def test_evm2():
    iq = np.load('/home/gaspar/git/ieee80211phy/data/sym8_rate24.npy')
    i = packet_detector(iq)[0]
    packet = receiver(iq[i - 2:])

    evm = evm_db2(*packet.data_symbols)
    assert int(evm) == -19


def test_evm3():
    """ This one has weird Sin shape EVM vs time.. sampling errro? """
    iq = np.load('/home/gaspar/git/ieee80211phy/data/sym130_rate24.npy')
    i = packet_detector(iq)[0]
    packet = receiver(iq[i - 2:])

    evm = evm_db2(*packet.data_symbols)
    assert int(evm) == -18


def test_evm4():
    """ This one has weird Sin shape EVM vs time.. sampling errro? """
    iq = np.load('/home/gaspar/git/ieee80211phy/data/sym173_rate18.npy')
    i = packet_detector(iq)[0]
    packet = receiver(iq[i - 3:])

    evm = evm_db2(*packet.data_symbols)
    assert int(evm) == -18


def test_evm5():
    iq = np.load('/home/gaspar/git/ieee80211phy/data/sym16_rate48.npy')
    i = packet_detector(iq)[0]
    packet = receiver(iq[i - 3:])

    evm = evm_db2(*packet.data_symbols)
    assert int(evm) == -19

# def test_limemini_lime_air():
#     """
#     Lime-Mini as TX and Lime as RX. First test with different devices, resulted in frequency offset - that
#     was best corrected using the pilot symbols!
#     """
#
#     iq = np.load('/home/gaspar/git/ieee80211phy/data/limemini_lime_air.npy')
#     r = Receiver(sample_advance=-3)
#     symbols = r.main(iq, n_symbols=229)
#
#     from ieee80211phy.transmitter.subcarrier_modulation_mapping import mapper_decide
#     reference_symbols = np.array([[mapper_decide(j, 4) for j in x] for x in symbols])
#     evm = evm_db(symbols, reference_symbols)
#     assert int(evm) == -26
