import logging
import numpy as np
import pytest

from ieee80211phy import scrambler
from ieee80211phy.conv_coding import conv_decode
from ieee80211phy.interleaving import interleave
from ieee80211phy.modulation import symbols_to_bits
from ieee80211phy.ofdm import demodulate_ofdm
from ieee80211phy.preamble import long_training_symbol
from ieee80211phy.scrambler import scrambler
from ieee80211phy.signal_field import decode_signal_field
from ieee80211phy.transmitter import get_params_from_rate, transmitter
from ieee80211phy.util import moving_average, hex_to_bitstr, awgn

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
                timings.append(i + 31)
            state = 0

    logger.info(f'Found packets at: {timings}')
    return timings


Trace = {}


def receiver(iq):
    """ Channel estimation - calculate how much the known symbols have changed and produce inverse channel """
    avg_train = (iq[:64] + iq[64:128]) / 2
    channel_estimate = np.fft.fft(avg_train) / long_training_symbol()
    equalizer = 1 / channel_estimate
    Trace['equalizer'] = equalizer

    """ Signal field demodulation - this gives us the data rate for the payload and also the payload length """
    signal = iq[128: 128 + 80]
    symbols = demodulate_ofdm(signal, equalizer, index_in_package=0)
    Trace['signal_symbols'] = symbols
    bits = symbols_to_bits(symbols, bits_per_symbol=1)
    bits = interleave(bits, coded_bits_symbol=48, coded_bits_subcarrier=1, undo=True)
    bits = conv_decode(bits)
    data_rate, length_bytes = decode_signal_field(bits)
    modulation, coding_rate, coded_bits_subcarrier, coded_bits_symbol, data_bits_symbol = get_params_from_rate(
        data_rate)
    n_ofdm_symbols = int(np.ceil((length_bytes * 8 + 22) / data_bits_symbol))
    logger.info(f'Package {length_bytes} bytes, {n_ofdm_symbols} OFDM symbols\n'
                f'\t data_rate={data_rate}, modulation={modulation}, coding_rate={coding_rate}')

    """ Payload demodulation """
    signal_end = 128 + 80
    data_groups = iq[signal_end: signal_end + (80 * n_ofdm_symbols)].reshape((-1, 80))
    ofdm_symbols = [demodulate_ofdm(group, equalizer, index_in_package=1 + i)
                    for i, group in enumerate(data_groups)]
    Trace['data_symbols'] = ofdm_symbols

    """ Symbols to bits flow """
    bits = [symbols_to_bits(symbol, bits_per_symbol=coded_bits_subcarrier)
            for symbol in ofdm_symbols]
    bits = ''.join(interleave(b, coded_bits_symbol, coded_bits_subcarrier, undo=True) for b in bits)
    bits = conv_decode(bits, coding_rate)
    bits = scrambler(bits)
    return bits[16:16 + length_bytes * 8]


@pytest.mark.parametrize('data_rate', [6, 9, 12, 18, 24, 36, 48, 54])
def test_loopback(data_rate):
    np.random.seed(0)
    tx = '0x0402002E006008CD37A60020D6013CF1006008AD3BAF00004A6F792C2062726967687420737061726B206F6620646976696E6974792C0A4461756768746572206F6620456C797369756D2C0A466972652D696E73697265642077652074726561673321B6'
    tx_bits = hex_to_bitstr(tx)
    iq = transmitter(tx_bits, data_rate)
    iq = awgn(iq, 20)

    i = packet_detector(iq)[0]
    rx_bits = receiver(iq[i - 2:])
    assert rx_bits == tx_bits


def test_packet_detector():
    iq = np.load('/home/gaspar/git/ieee80211phy/data/limemini_wire_loopback.npy')
    iq = np.hstack([iq, iq, iq, iq])
    indexes = packet_detector(iq)
    assert indexes == [95058, 295058, 495058, 695058]
