import os
from typing import List

import numpy as np
from hypothesis import settings, given
from hypothesis._strategies import composite, integers, binary, sampled_from

from wifi import convolutional_coder, header, interleaver, modulator, scrambler, bits, puncturer, preambler, \
    padder, subcarrier_mapping, pilots, to_time_domain, guard_interval, merger, channel_impairments, bitstr
from wifi.config import Config
from loguru import logger


def do(data: bits, data_rate: int):
    data_len_bytes = len(bitstr.split(data, 8))
    conf = Config.from_data_rate(data_rate)

    """ Bit-domain """
    data, n_pad = padder.do(data, conf.data_bits_per_ofdm_symbol)
    data = scrambler.do(data)
    data = data[:-n_pad - 6] + '000000' + data[-n_pad:]  # Refer to 17.3.5.3 for details.
    data = convolutional_coder.do(data)
    data = puncturer.do(data, conf.coding_rate)
    data = interleaver.do(data, conf.coded_bits_per_ofdm_symbol, conf.coded_bits_per_carrier_symbol)
    symbols = header.do(data_rate, data_len_bytes) + modulator.do(data, conf.coded_bits_per_carrier_symbol)

    """ Frequency-domain """
    carriers = subcarrier_mapping.do(symbols)
    carriers = pilots.do(carriers)
    frames = to_time_domain.do(carriers)

    """ Time-domain """
    frames = guard_interval.do(frames)
    result = merger.do(preambler.short_training_sequence(), preambler.long_training_sequence(), frames)

    logger.info(f'{data_len_bytes}B @ {data_rate}MB/s ({conf.modulation}, {conf.coding_rate})')
    return result


def undo(iq: List[complex]) -> bits:
    """ Time-domain """
    # iq = sampling_offset.undo(iq, bias=-2)
    # iq = frequency_offset.undo(iq)
    short_training, long_training, frames = merger.undo(iq)
    frames = guard_interval.undo(frames)
    carriers = to_time_domain.undo(frames)

    """ Frequency-domain """
    carriers = channel_impairments.undo(long_training, carriers)
    carriers = pilots.undo(carriers)
    symbols = subcarrier_mapping.undo(carriers)
    data_rate, length_bytes = header.undo(symbols[:48])
    conf = Config.from_data_rate(data_rate)
    n_ofdm_symbols = int(np.ceil((length_bytes * 8 + 22) / conf.data_bits_per_ofdm_symbol))
    symbols = symbols[48:48*(n_ofdm_symbols + 1)]
    data = modulator.undo(symbols, bits_per_symbol=conf.coded_bits_per_carrier_symbol)

    """ Bit-domain """
    data = interleaver.undo(data, conf.coded_bits_per_ofdm_symbol, conf.coded_bits_per_carrier_symbol)
    data = puncturer.undo(data, conf.coding_rate)
    data = convolutional_coder.undo(data)
    data = scrambler.undo(data)
    data = padder.undo(data, length_bytes)

    logger.info(f'{length_bytes}B @ {data_rate}MB/s ({conf.modulation}, {conf.coding_rate})')
    return data


def test_annexi():
    """ This is the full test-case provided in the WiFi standard (paragraph ANNEX I) """
    # Table I-1—The message for the BCC example - i have reversed bit ordering in each byte
    input = bitstr.from_hex(
        '0x20400074000610b3ec6500046b803c8f000610b5dcf5000052f69e3404464e96e6162e04ce0e864ed604f6660426966e967'
        '6962e9e34502286aee6162ea64e04f66604a2369ece96aeb6345062964ea6b49676ce964ea62604eea6042e4ea686e6cc846d'
    )

    dir_path = os.path.dirname(os.path.realpath(__file__)) + '/'
    expected = np.load(dir_path + '../data/ieee_full_packet_test_case.npy')

    output = do(input, data_rate=36)
    np.testing.assert_equal(np.round(output, 3), expected)

    undo_bits = undo(output)
    assert input == undo_bits


@composite
def random_packet(draw):
    elements = draw(integers(min_value=0, max_value=(2**12)-1))
    data = draw(binary(min_size=elements, max_size=elements))
    data = bitstr.from_bytes(data)
    rate = draw(sampled_from([6, 9, 12, 18, 24, 36, 48, 54]))
    return data, rate


@settings(deadline=None)
@given(random_packet())
def test_hypothesis(data_test):
    data, rate = data_test
    assert undo(do(data, data_rate=rate)) == data
