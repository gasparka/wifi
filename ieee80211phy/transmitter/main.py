from textwrap import wrap

import numpy as np

from ieee80211phy.transmitter.convolutional_encoder import convolutional_encoder
from ieee80211phy.transmitter.interleaver import interleaver
from ieee80211phy.transmitter.ofdm_modulation import map_to_carriers, insert_pilots, ifft_guard
from ieee80211phy.transmitter.preamble import preamble
from ieee80211phy.transmitter.scrambler import scrambler
from ieee80211phy.transmitter.signal_field import signal_field
from ieee80211phy.transmitter.subcarrier_modulation_mapping import mapper


def get_params_from_rate(data_rate):
    """
    Table 17-4—Modulation-dependent parameters
    +------------+---------+----------------+-----------------+-----------------+--------------+
    | Modulation |         |                |                 |                 |              |
    |            | Coding  |   Coded bits   |   Coded bits    |    Data bits    |   Data Rate  |
    |            |         |                |                 |                 | (20MHz band) |
    |            |   rate  | per subcarrier | per OFDM symbol | per OFDM symbol |              |
    +------------+---------+----------------+-----------------+-----------------+--------------+
    |    BPSK    |   1/2   |        1       |        48       |        24       |       6      |
    +------------+---------+----------------+-----------------+-----------------+--------------+
    |    BPSK    |   3/4   |        1       |        48       |        36       |       9      |
    +------------+---------+----------------+-----------------+-----------------+--------------+
    |    QPSK    |   1/2   |        2       |        96       |        48       |      12      |
    +------------+---------+----------------+-----------------+-----------------+--------------+
    |    QPSK    |   3/4   |        2       |        96       |        72       |      18      |
    +------------+---------+----------------+-----------------+-----------------+--------------+
    |   16-QAM   |   1/2   |        4       |       192       |        96       |      24      |
    +------------+---------+----------------+-----------------+-----------------+--------------+
    |   16-QAM   |   3/4   |        4       |       192       |       144       |      36      |
    +------------+---------+----------------+-----------------+-----------------+--------------+
    |   64-QAM   |   2/3   |        6       |       288       |       192       |      48      |
    +------------+---------+----------------+-----------------+-----------------+--------------+
    |   64-QAM   |   3/4   |        6       |       288       |       216       |      54      |
    +------------+---------+----------------+-----------------+-----------------+--------------+
    """
    if data_rate == 6:
        return 'BPSK',      '1/2',  1,  48,     24
    elif data_rate == 9:
        return 'BPSK',      '3/4',  1,  48,     36
    elif data_rate == 12:
        return 'QPSK',      '1/2',  2,  96,     48
    elif data_rate == 18:
        return 'QPSK',      '3/4',  2,  96,     72
    elif data_rate == 24:
        return '16-QAM',    '1/2',  4,  192,    96
    elif data_rate == 36:
        return '16-QAM',    '3/4',  4,  192,    144
    elif data_rate == 48:
        return '64-QAM',    '2/3',  6,  288,    192
    elif data_rate == 54:
        return '64-QAM',    '3/4',  6,  288,    216


def bytes_count(data):
    if data[0:2] in ('0x', '0X'):
        data = data[2:]

    length = len(data)
    assert not length & 1  # is divisible by 2
    return length // 2


def build_package(data, data_rate):
    """
    a) Produce the PHY Preamble field, composed of 10 repetitions of a “short training sequence” (used
    for AGC convergence, diversity selection, timing acquisition, and coarse frequency acquisition in
    the receiver) and two repetitions of a “long training sequence” (used for channel estimation and fine
    frequency acquisition in the receiver), preceded by a guard interval (GI). Refer to 17.3.3 for details.
    """
    pre = preamble()

    """
    b) Produce the PHY header field from the RATE, LENGTH fields. In order to facilitate a reliable and timely
    detection of the RATE and LENGTH fields, 6 zero tail bits are inserted into the PHY header. 
    
    The encoding of the SIGNAL field into an OFDM symbol follows the same steps for convolutional
    encoding, interleaving, BPSK modulation, pilot insertion, Fourier transform, and prepending a GI as
    described subsequently for data transmission with BPSK-OFDM modulated at coding rate 1/2. The
    contents of the SIGNAL field are not scrambled. Refer to 17.3.4 for details.
    """
    header_bits = signal_field(data_rate, length_bytes=len(wrap(data, 8)))
    header_conv = convolutional_encoder(header_bits, '1/2')
    header_interleav = interleaver(header_conv, coded_bits_symbol=48, coded_bits_subcarrier=1)
    header_mapped = mapper(header_interleav, bits_per_symbol=1)
    header_symbols = map_to_carriers(header_mapped)
    header_symbols = insert_pilots(header_symbols, 0)
    header_time_domain = ifft_guard(header_symbols)

    """
    c) Calculate from RATE field of the TXVECTOR the number of data bits per OFDM symbol (N DBPS ),
    the coding rate (R), the number of bits in each OFDM subcarrier (N BPSC ), and the number of coded
    bits per OFDM symbol (N CBPS ). Refer to 17.3.2.3 for details.
    """
    modulation, coding_rate, coded_bits_subcarrier, \
    coded_bits_symbol, data_bits_symbol = get_params_from_rate(data_rate)

    """
    d) Append the PSDU to the SERVICE field of the TXVECTOR. Extend the resulting bit string with
    zero bits (at least 6 bits) so that the resulting length is a multiple of N DBPS . The resulting bit string
    constitutes the DATA part of the packet. Refer to 17.3.5.4 for details.
    
    The SERVICE field has 16 bits. The bits from 0–6 of the SERVICE field, which are transmitted first, are set to 0s and are used to
    synchronize the descrambler in the receiver. The remaining 9 bits (7–15) of the SERVICE field shall be
    reserved for future use. All reserved bits shall be set to 0 on transmission and ignored on reception. Refer to
    Figure 17-6.
    
    The PPDU TAIL field shall be six bits of 0, which are required to return the convolutional encoder to the
    zero state. This procedure improves the error probability of the convolutional decoder, which relies on future
    bits when decoding and which may be not be available past the end of the message.
    """
    service = '0' * 16
    tail = '0' * 6
    data = service + data + tail

    n_symbols = int(np.ceil(len(data) / data_bits_symbol))
    n_data = n_symbols * data_bits_symbol
    n_pad = int(n_data - len(data))
    pad = '0' * n_pad

    data = data + pad

    """
    e) If the TXVECTOR parameter CH_BANDWIDTH_IN_NON_HT is not present, initiate the
    scrambler with a pseudorandom nonzero seed and generate a scrambling sequence. 
    XOR the scrambling sequence with the extended string of data bits. Refer to 17.3.5.5 for details.
    """
    data = scrambler(data)

    """ 
    f) Replace the six scrambled zero bits following the data with six nonscrambled zero bits.
    (Those bits return the convolutional encoder to the zero state and are denoted as tail bits.) 
    Refer to 17.3.5.3 for details. 
    """
    data = data[:-len(pad)-6] + '000000' + data[-len(pad):]

    """
    g) Encode the extended, scrambled data string with a convolutional encoder (R = 1/2). Omit (puncture)
    some of the encoder output string (chosen according to “puncturing pattern”) to reach the “coding
    rate” corresponding to the TXVECTOR parameter RATE. Refer to 17.3.5.6 for details.
    """
    data = convolutional_encoder(data, coding_rate)

    """
    h) Divide the encoded bit string into groups of 'coded_bits_symbol' bits. Within each group, perform an
    “interleaving” (reordering) of the bits according to a rule corresponding to the TXVECTOR
    parameter RATE. Refer to 17.3.5.7 for details.
    """
    groups = wrap(data, coded_bits_symbol)
    interleaved_groups = [interleaver(group, coded_bits_symbol, coded_bits_subcarrier) for group in groups]
    data = ''.join(interleaved_groups)

    """
    i) Divide the resulting coded and interleaved data string into groups of 'coded_bits_subcarrier' bits. For each of the bit
    groups, convert the bit group into a complex number according to the modulation encoding tables.
    Refer to 17.3.5.8 for details.
    """
    groups = wrap(data, coded_bits_subcarrier)
    data_complex = np.array([mapper(group, coded_bits_subcarrier) for group in groups])

    """
    j) Divide the complex number string into groups of 48 complex numbers. Each such group is
    associated with one OFDM symbol. In each group, the complex numbers are numbered 0 to 47 and
    mapped hereafter into OFDM subcarriers numbered –26 to –22, –20 to –8, –6 to –1, 1 to 6, 8 to 20,
    and 22 to 26. The subcarriers –21, –7, 7, and 21 are skipped and, subsequently, used for inserting
    pilot subcarriers. The 0 subcarrier, associated with center frequency, is omitted and filled with the
    value 0. Refer to 17.3.5.10 for details.
    """
    ofdm_symbols = np.reshape(data_complex, (-1, 48))
    ofdm_symbols = [map_to_carriers(symbol) for symbol in ofdm_symbols]

    """
    k) Four subcarriers are inserted as pilots into positions –21, –7, 7, and 21.
    Refer to 17.3.5.9 for details.
    """
    ofdm_symbols = [insert_pilots(symbol, symbol_i + 1) for symbol_i, symbol in enumerate(ofdm_symbols)]

    """
    l) For each group of subcarriers –26 to 26, convert the subcarriers to time domain using inverse
    Fourier transform. Prepend to the Fourier-transformed waveform a circular extension of itself thus
    forming a GI, and truncate the resulting periodic waveform to a single OFDM symbol length by
    applying time domain windowing. Refer to 17.3.5.10 for details.
    """
    data_time_domain = [ifft_guard(symbol) for symbol in ofdm_symbols]

    for i in range(1, len(data_time_domain)):
        data_time_domain[i][0] = (data_time_domain[i-1][-64] + data_time_domain[i][0]) / 2

    data_time_domain = np.concatenate(data_time_domain)

    """
    m) Append the OFDM symbols one after another, starting after the SIGNAL symbol describing the
    RATE and LENGTH fields. Refer to 17.3.5.10 for details.
    """
    #     # time windowing method as discussed in "17.3.2.6 Discrete time implementation considerations"
    # merge preamble with header
    # pre[-1] = (pre[-1] + header_time_domain[0]) / 2
    header_time_domain[0] = (pre[-64] + header_time_domain[0]) / 2
    data_time_domain[0] = (header_time_domain[-64] + data_time_domain[0]) / 2
    result = np.concatenate([pre, header_time_domain, data_time_domain, [data_time_domain[-64]/2]])
    return result
    pass
    # return sig


def hex_to_bitstr(hstr):
    """ http://stackoverflow.com/questions/1425493/convert-hex-to-binary """
    assert isinstance(hstr, str)
    if hstr[0:2] in ('0x', '0X'):
        hstr = hstr[2:]
    my_hexdata = hstr
    scale = 16  ## equals to hexadecimal
    num_of_bits = int(len(my_hexdata) * np.log2(scale))
    return bin(int(my_hexdata, scale))[2:].zfill(num_of_bits)

def flip_byte_endian(bitstr):
    from textwrap import wrap
    bytes = wrap(bitstr, 8)
    flipped = [x[::-1] for x in bytes]
    return ''.join(flipped)

def test_():
    # Table I-1—The message for the BCC example
    data = '0x0402002E006008CD37A60020D6013CF1006008AD3BAF00004A6F792C2062726967687420737061726B206F6620646976696E6974792C0A4461756768746572206F6620456C797369756D2C0A466972652D696E73697265642077652074726561673321B6'
    bin = flip_byte_endian(hex_to_bitstr(data))

    package_iq = build_package(bin, data_rate=36)
    pass

# 0402002E006008CD37A60020D6013CF1006008AD3BAF00004A6F792C2062726967687420737061726B206F6620646976696E6974792C0A4461756768746572206F6620456C797369756D2C0A466972652D696E73697265642077652074726561673321B6
#
#
# def get_preamble():
#     short = get_short_training()
#     long = get_long()
#     short[-1] += long[0]
#
#     result = np.concatenate([short, long[1:]])
#     return result
#
#
# def get_signal_field():
#     # TODO: this is currently hardcoded for the IEE example i.e.
#     # Table I-9—SIGNAL field bits after interleaving
#     input = '100101001101000000010100100000110010010010010100'
#     input = np.array([1 if x == '1' else 0 for x in input])
#     symbols = mapper(input, bits_per_symbol=1)
#     symbols = inset_pilots_and_pad(symbols)
#     time_domain = ifft_guard_window(symbols)
#     return time_domain
#
#
# def get_data_field(bits=None):
#     bits = '011101111111000011101111110001000111001100000000101111110001000100010000100110100001110100010010011011100011100011110101011010010001101101101011100110000100001100000000000011011011001101101101'
#     input = np.array([1 if x == '1' else 0 for x in bits])
#     symbols = mapper(input, bits_per_symbol=4)
#     symbols = inset_pilots_and_pad(symbols)
#     time_domain = ifft_guard_window(symbols)
#     return time_domain
#
#
# def make_tx():
#     preamble = get_preamble()
#     signal = get_signal_field()
#     preamble[-1] += signal[0]
#
#     data1 = get_data_field()
#     signal[-1] += data1[0]
#
#     result = np.concatenate([preamble, signal[1:], data1[1:]])
#     return result
#
#
# def test_():
#     # Table I-22—Time domain representation of the short training sequence
#     expect_short_train = [(0.023 + 0.023j), (-0.132 + 0.002j), (-0.013 - 0.079j), (0.143 - 0.013j), (0.092 + 0j),
#                           (0.143 - 0.013j), (-0.013 - 0.079j), (-0.132 + 0.002j), (0.046 + 0.046j), (0.002 - 0.132j),
#                           (-0.079 - 0.013j), (-0.013 + 0.143j), 0.092j, (-0.013 + 0.143j), (-0.079 - 0.013j),
#                           (0.002 - 0.132j), (0.046 + 0.046j), (-0.132 + 0.002j), (-0.013 - 0.079j), (0.143 - 0.013j),
#                           (0.092 + 0j), (0.143 - 0.013j), (-0.013 - 0.079j), (-0.132 + 0.002j), (0.046 + 0.046j),
#                           (0.002 - 0.132j), (-0.079 - 0.013j), (-0.013 + 0.143j), 0.092j, (-0.013 + 0.143j),
#                           (-0.079 - 0.013j), (0.002 - 0.132j), (0.046 + 0.046j), (-0.132 + 0.002j), (-0.013 - 0.079j),
#                           (0.143 - 0.013j), (0.092 + 0j), (0.143 - 0.013j), (-0.013 - 0.079j), (-0.132 + 0.002j),
#                           (0.046 + 0.046j), (0.002 - 0.132j), (-0.079 - 0.013j), (-0.013 + 0.143j), 0.092j,
#                           (-0.013 + 0.143j), (-0.079 - 0.013j), (0.002 - 0.132j), (0.046 + 0.046j), (-0.132 + 0.002j),
#                           (-0.013 - 0.079j), (0.143 - 0.013j), (0.092 + 0j), (0.143 - 0.013j), (-0.013 - 0.079j),
#                           (-0.132 + 0.002j), (0.046 + 0.046j), (0.002 - 0.132j), (-0.079 - 0.013j), (-0.013 + 0.143j),
#                           0.092j, (-0.013 + 0.143j), (-0.079 - 0.013j), (0.002 - 0.132j), (0.046 + 0.046j),
#                           (-0.132 + 0.002j), (-0.013 - 0.079j), (0.143 - 0.013j), (0.092 + 0j), (0.143 - 0.013j),
#                           (-0.013 - 0.079j), (-0.132 + 0.002j), (0.046 + 0.046j), (0.002 - 0.132j), (-0.079 - 0.013j),
#                           (-0.013 + 0.143j), 0.092j, (-0.013 + 0.143j), (-0.079 - 0.013j), (0.002 - 0.132j),
#                           (0.046 + 0.046j), (-0.132 + 0.002j), (-0.013 - 0.079j), (0.143 - 0.013j), (0.092 + 0j),
#                           (0.143 - 0.013j), (-0.013 - 0.079j), (-0.132 + 0.002j), (0.046 + 0.046j), (0.002 - 0.132j),
#                           (-0.079 - 0.013j), (-0.013 + 0.143j), 0.092j, (-0.013 + 0.143j), (-0.079 - 0.013j),
#                           (0.002 - 0.132j), (0.046 + 0.046j), (-0.132 + 0.002j), (-0.013 - 0.079j), (0.143 - 0.013j),
#                           (0.092 + 0j), (0.143 - 0.013j), (-0.013 - 0.079j), (-0.132 + 0.002j), (0.046 + 0.046j),
#                           (0.002 - 0.132j), (-0.079 - 0.013j), (-0.013 + 0.143j), 0.092j, (-0.013 + 0.143j),
#                           (-0.079 - 0.013j), (0.002 - 0.132j), (0.046 + 0.046j), (-0.132 + 0.002j), (-0.013 - 0.079j),
#                           (0.143 - 0.013j), (0.092 + 0j), (0.143 - 0.013j), (-0.013 - 0.079j), (-0.132 + 0.002j),
#                           (0.046 + 0.046j), (0.002 - 0.132j), (-0.079 - 0.013j), (-0.013 + 0.143j), 0.092j,
#                           (-0.013 + 0.143j), (-0.079 - 0.013j), (0.002 - 0.132j), (0.046 + 0.046j), (-0.132 + 0.002j),
#                           (-0.013 - 0.079j), (0.143 - 0.013j), (0.092 + 0j), (0.143 - 0.013j), (-0.013 - 0.079j),
#                           (-0.132 + 0.002j), (0.046 + 0.046j), (0.002 - 0.132j), (-0.079 - 0.013j), (-0.013 + 0.143j),
#                           0.092j, (-0.013 + 0.143j), (-0.079 - 0.013j), (0.002 - 0.132j), (0.046 + 0.046j),
#                           (-0.132 + 0.002j), (-0.013 - 0.079j), (0.143 - 0.013j), (0.092 + 0j), (0.143 - 0.013j),
#                           (-0.013 - 0.079j), (-0.132 + 0.002j), (0.046 + 0.046j), (0.002 - 0.132j), (-0.079 - 0.013j),
#                           (-0.013 + 0.143j), 0.092j, (-0.013 + 0.143j), (-0.079 - 0.013j), (0.002 - 0.132j)]
#
#     # Table I-23—Time domain representation of the long training sequence
#     expect_long_train = [(-0.055 + 0.023j), (0.012 - 0.098j), (0.092 - 0.106j), (-0.092 - 0.115j), (-0.003 - 0.054j),
#                          (0.075 + 0.074j), (-0.127 + 0.021j), (-0.122 + 0.017j), (-0.035 + 0.151j), (-0.056 + 0.022j),
#                          (-0.06 - 0.081j), (0.07 - 0.014j), (0.082 - 0.092j), (-0.131 - 0.065j), (-0.057 - 0.039j),
#                          (0.037 - 0.098j), (0.062 + 0.062j), (0.119 + 0.004j), (-0.022 - 0.161j), (0.059 + 0.015j),
#                          (0.024 + 0.059j), (-0.137 + 0.047j), (0.001 + 0.115j), (0.053 - 0.004j), (0.098 + 0.026j),
#                          (-0.038 + 0.106j), (-0.115 + 0.055j), (0.06 + 0.088j), (0.021 - 0.028j), (0.097 - 0.083j),
#                          (0.04 + 0.111j), (-0.005 + 0.12j), (0.156 + 0j), (-0.005 - 0.12j), (0.04 - 0.111j),
#                          (0.097 + 0.083j), (0.021 + 0.028j), (0.06 - 0.088j), (-0.115 - 0.055j), (-0.038 - 0.106j),
#                          (0.098 - 0.026j), (0.053 + 0.004j), (0.001 - 0.115j), (-0.137 - 0.047j), (0.024 - 0.059j),
#                          (0.059 - 0.015j), (-0.022 + 0.161j), (0.119 - 0.004j), (0.062 - 0.062j), (0.037 + 0.098j),
#                          (-0.057 + 0.039j), (-0.131 + 0.065j), (0.082 + 0.092j), (0.07 + 0.014j), (-0.06 + 0.081j),
#                          (-0.056 - 0.022j), (-0.035 - 0.151j), (-0.122 - 0.017j), (-0.127 - 0.021j), (0.075 - 0.074j),
#                          (-0.003 + 0.054j), (-0.092 + 0.115j), (0.092 + 0.106j), (0.012 + 0.098j), (-0.156 + 0j),
#                          (0.012 - 0.098j), (0.092 - 0.106j), (-0.092 - 0.115j), (-0.003 - 0.054j), (0.075 + 0.074j),
#                          (-0.127 + 0.021j), (-0.122 + 0.017j), (-0.035 + 0.151j), (-0.056 + 0.022j), (-0.06 - 0.081j),
#                          (0.07 - 0.014j), (0.082 - 0.092j), (-0.131 - 0.065j), (-0.057 - 0.039j), (0.037 - 0.098j),
#                          (0.062 + 0.062j), (0.119 + 0.004j), (-0.022 - 0.161j), (0.059 + 0.015j), (0.024 + 0.059j),
#                          (-0.137 + 0.047j), (0.001 + 0.115j), (0.053 - 0.004j), (0.098 + 0.026j), (-0.038 + 0.106j),
#                          (-0.115 + 0.055j), (0.06 + 0.088j), (0.021 - 0.028j), (0.097 - 0.083j), (0.04 + 0.111j),
#                          (-0.005 + 0.12j), (0.156 + 0j), (-0.005 - 0.12j), (0.04 - 0.111j), (0.097 + 0.083j),
#                          (0.021 + 0.028j), (0.06 - 0.088j), (-0.115 - 0.055j), (-0.038 - 0.106j), (0.098 - 0.026j),
#                          (0.053 + 0.004j), (0.001 - 0.115j), (-0.137 - 0.047j), (0.024 - 0.059j), (0.059 - 0.015j),
#                          (-0.022 + 0.161j), (0.119 - 0.004j), (0.062 - 0.062j), (0.037 + 0.098j), (-0.057 + 0.039j),
#                          (-0.131 + 0.065j), (0.082 + 0.092j), (0.07 + 0.014j), (-0.06 + 0.081j), (-0.056 - 0.022j),
#                          (-0.035 - 0.151j), (-0.122 - 0.017j), (-0.127 - 0.021j), (0.075 - 0.074j), (-0.003 + 0.054j),
#                          (-0.092 + 0.115j), (0.092 + 0.106j), (0.012 + 0.098j), (-0.156 + 0j), (0.012 - 0.098j),
#                          (0.092 - 0.106j), (-0.092 - 0.115j), (-0.003 - 0.054j), (0.075 + 0.074j), (-0.127 + 0.021j),
#                          (-0.122 + 0.017j), (-0.035 + 0.151j), (-0.056 + 0.022j), (-0.06 - 0.081j), (0.07 - 0.014j),
#                          (0.082 - 0.092j), (-0.131 - 0.065j), (-0.057 - 0.039j), (0.037 - 0.098j), (0.062 + 0.062j),
#                          (0.119 + 0.004j), (-0.022 - 0.161j), (0.059 + 0.015j), (0.024 + 0.059j), (-0.137 + 0.047j),
#                          (0.001 + 0.115j), (0.053 - 0.004j), (0.098 + 0.026j), (-0.038 + 0.106j), (-0.115 + 0.055j),
#                          (0.06 + 0.088j), (0.021 - 0.028j), (0.097 - 0.083j), (0.04 + 0.111j), (-0.005 + 0.12j)]
#
#     # Table I-24—Time domain representation of the SIGNAL field (1 symbol)
#     expect_signal_field = [(0.109 + 0j), (0.033 - 0.044j), (-0.002 - 0.038j), (-0.081 + 0.084j), (0.007 - 0.1j),
#                            (-0.001 - 0.113j), (-0.021 - 0.005j), (0.136 - 0.105j), (0.098 - 0.044j), (0.011 - 0.002j),
#                            (-0.033 + 0.044j), (-0.06 + 0.124j), (0.01 + 0.097j), -0.008j, (0.018 - 0.083j),
#                            (-0.069 + 0.027j), (-0.219 + 0j), (-0.069 - 0.027j), (0.018 + 0.083j), 0.008j,
#                            (0.01 - 0.097j), (-0.06 - 0.124j), (-0.033 - 0.044j), (0.011 + 0.002j), (0.098 + 0.044j),
#                            (0.136 + 0.105j), (-0.021 + 0.005j), (-0.001 + 0.113j), (0.007 + 0.1j), (-0.081 - 0.084j),
#                            (-0.002 + 0.038j), (0.033 + 0.044j), (0.062 + 0j), (0.057 + 0.052j), (0.016 + 0.174j),
#                            (0.035 + 0.116j), (-0.051 - 0.202j), (0.011 + 0.036j), (0.089 + 0.209j), (-0.049 - 0.008j),
#                            (-0.035 + 0.044j), (0.017 - 0.059j), (0.053 - 0.017j), (0.099 + 0.1j), (0.034 - 0.148j),
#                            (-0.003 - 0.094j), (-0.12 + 0.042j), (-0.136 - 0.07j), (-0.031 + 0j), (-0.136 + 0.07j),
#                            (-0.12 - 0.042j), (-0.003 + 0.094j), (0.034 + 0.148j), (0.099 - 0.1j), (0.053 + 0.017j),
#                            (0.017 + 0.059j), (-0.035 - 0.044j), (-0.049 + 0.008j), (0.089 - 0.209j), (0.011 - 0.036j),
#                            (-0.051 + 0.202j), (0.035 - 0.116j), (0.016 - 0.174j), (0.057 - 0.052j), (0.062 + 0j),
#                            (0.033 - 0.044j), (-0.002 - 0.038j), (-0.081 + 0.084j), (0.007 - 0.1j), (-0.001 - 0.113j),
#                            (-0.021 - 0.005j), (0.136 - 0.105j), (0.098 - 0.044j), (0.011 - 0.002j), (-0.033 + 0.044j),
#                            (-0.06 + 0.124j), (0.01 + 0.097j), -0.008j, (0.018 - 0.083j), (-0.069 + 0.027j)]
#
#     expected = np.concatenate([expect_short_train, expect_short_train, expect_signal_field])
#     output = np.round(make_tx(), 3)
#
#     print(expected.shape, output.shape)
#     np.testing.assert_equal(expected, output[:len(expected)])
#
#     offset = 0
#     np.testing.assert_equal(expect_short_train, output[offset:len(expect_short_train)])
#     offset += len(expect_short_train)
#
#     np.testing.assert_equal(expect_long_train,
#                             output[offset: offset + len(expect_long_train)])
#     offset += len(expect_long_train)
#
#     # np.testing.assert_equal(expect_signal_field,
#     #                         output[offset: offset + len(expect_signal_field)])
#     # offset += len(expect_long_train)
#
#     pass
