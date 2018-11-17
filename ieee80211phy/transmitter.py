import logging
from textwrap import wrap
import numpy as np

from ieee80211phy.conv_coding import conv_encode
from ieee80211phy.interleaving import interleave
from ieee80211phy.modulation import bits_to_symbols
from ieee80211phy.ofdm import modulate_ofdm
from ieee80211phy.preamble import short_training_sequence, long_training_sequence
from ieee80211phy.scrambler import scrambler
from ieee80211phy.signal_field import encode_signal_field

log = logging.getLogger(__name__)


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
        return 'BPSK', '1/2', 1, 48, 24
    elif data_rate == 9:
        return 'BPSK', '3/4', 1, 48, 36
    elif data_rate == 12:
        return 'QPSK', '1/2', 2, 96, 48
    elif data_rate == 18:
        return 'QPSK', '3/4', 2, 96, 72
    elif data_rate == 24:
        return '16-QAM', '1/2', 4, 192, 96
    elif data_rate == 36:
        return '16-QAM', '3/4', 4, 192, 144
    elif data_rate == 48:
        return '64-QAM', '2/3', 6, 288, 192
    elif data_rate == 54:
        return '64-QAM', '3/4', 6, 288, 216


def transmitter(data, data_rate):
    """
    a) Produce the PHY Preamble field, composed of 10 repetitions of a “short training sequence” (used
    for AGC convergence, diversity selection, timing acquisition, and coarse frequency acquisition in
    the receiver) and two repetitions of a “long training sequence” (used for channel estimation and fine
    frequency acquisition in the receiver), preceded by a guard interval (GI). Refer to 17.3.3 for details.
    """
    train_short = short_training_sequence()
    train_long = long_training_sequence()

    """
    b) Produce the PHY header field from the RATE, LENGTH fields. In order to facilitate a reliable and timely
    detection of the RATE and LENGTH fields, 6 zero tail bits are inserted into the PHY header. 
    
    The encoding of the SIGNAL field into an OFDM symbol follows the same steps for convolutional
    encoding, interleaving, BPSK modulation, pilot insertion, Fourier transform, and prepending a GI as
    described subsequently for data transmission with BPSK-OFDM modulated at coding rate 1/2. The
    contents of the SIGNAL field are not scrambled. Refer to 17.3.4 for details.
    """
    n_bytes = len(wrap(data, 8))
    signal = encode_signal_field(data_rate, length_bytes=n_bytes)
    signal = conv_encode(signal, '1/2')
    signal = interleave(signal, coded_bits_symbol=48, coded_bits_subcarrier=1)
    signal = bits_to_symbols(signal, bits_per_symbol=1)
    signal = modulate_ofdm(signal, index_in_package=0)

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
    log.info(f'Package {n_bytes} bytes, {n_symbols} OFDM symbols ({n_pad} padding bits added)\n'
             f'\t data_rate={data_rate}, modulation={modulation}, coding_rate={coding_rate}')

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
    data = data[:-len(pad) - 6] + '000000' + data[-len(pad):]

    """
    g) Encode the extended, scrambled data string with a convolutional encoder (R = 1/2). Omit (puncture)
    some of the encoder output string (chosen according to “puncturing pattern”) to reach the “coding
    rate” corresponding to the TXVECTOR parameter RATE. Refer to 17.3.5.6 for details.
    """
    data = conv_encode(data, coding_rate)

    """
    h) Divide the encoded bit string into groups of 'coded_bits_symbol' bits. Within each group, perform an
    “interleaving” (reordering) of the bits according to a rule corresponding to the TXVECTOR
    parameter RATE. Refer to 17.3.5.7 for details.
    """
    interleaved = [interleave(bit_group, coded_bits_symbol, coded_bits_subcarrier)
                          for bit_group in wrap(data, coded_bits_symbol)]
    data = ''.join(interleaved)

    """
    i) Divide the resulting coded and interleaved data string into groups of 'coded_bits_subcarrier' bits. 
    For each of the bit groups, convert the bit group into a complex number according to the modulation encoding tables.
    Refer to 17.3.5.8 for details.
    """
    data = bits_to_symbols(data, coded_bits_subcarrier)

    """
    j) Divide the complex number string into groups of 48 complex numbers. Each such group is
    associated with one OFDM symbol. In each group, the complex numbers are numbered 0 to 47 and
    mapped hereafter into OFDM subcarriers numbered –26 to –22, –20 to –8, –6 to –1, 1 to 6, 8 to 20,
    and 22 to 26. The subcarriers –21, –7, 7, and 21 are skipped and, subsequently, used for inserting
    pilot subcarriers. The 0 subcarrier, associated with center frequency, is omitted and filled with the
    value 0. Refer to 17.3.5.10 for details.

    k) Four subcarriers are inserted as pilots into positions –21, –7, 7, and 21.
    Refer to 17.3.5.9 for details.

    l) For each group of subcarriers –26 to 26, convert the subcarriers to time domain using inverse
    Fourier transform. Prepend to the Fourier-transformed waveform a circular extension of itself thus
    forming a GI, and truncate the resulting periodic waveform to a single OFDM symbol length by
    applying time domain windowing. Refer to 17.3.5.10 for details.
    """
    data = [modulate_ofdm(ofdm_symbol, index_in_package=i + 1)
            for i, ofdm_symbol in enumerate(data.reshape((-1, 48)))]

    """
    m) Append the OFDM symbols one after another, starting after the SIGNAL symbol describing the
    RATE and LENGTH fields. Refer to 17.3.5.10 for details.
    
    Note: also doing time domain windowing, as discussed in "17.3.2.6 Discrete time implementation considerations"
    """

    train_short[0] = train_short[0] / 2
    # merge short and long training sequence
    train_long[0] = (train_short[-64] + train_long[0]) / 2

    # merge long training sequence with header
    signal[0] = (train_long[-64] + signal[0]) / 2

    # merge header with data
    data[0][0] = (signal[-64] + data[0][0]) / 2

    # merge each data symbol
    for i in range(1, len(data)):
        data[i][0] = (data[i - 1][-64] + data[i][0]) / 2
    data = np.concatenate(data)

    result = np.concatenate([train_short, train_long, signal, data, [data[-64] / 2]])
    return result


def test_annexi():
    from ieee80211phy.util import flip_byte_endian, hex_to_bitstr
    # Table I-1—The message for the BCC example
    input = '0x0402002E006008CD37A60020D6013CF1006008AD3BAF00004A6F792C2062726967687420737061726B206F6620646976696E6974792C0A4461756768746572206F6620456C797369756D2C0A466972652D696E73697265642077652074726561673321B6'
    input = flip_byte_endian(hex_to_bitstr(input))

    # Table I-22—Time domain representation of the short training sequence
    expect_short_train = [(0.023 + 0.023j), (-0.132 + 0.002j), (-0.013 - 0.079j), (0.143 - 0.013j), (0.092 + 0j),
                          (0.143 - 0.013j), (-0.013 - 0.079j), (-0.132 + 0.002j), (0.046 + 0.046j), (0.002 - 0.132j),
                          (-0.079 - 0.013j), (-0.013 + 0.143j), 0.092j, (-0.013 + 0.143j), (-0.079 - 0.013j),
                          (0.002 - 0.132j), (0.046 + 0.046j), (-0.132 + 0.002j), (-0.013 - 0.079j), (0.143 - 0.013j),
                          (0.092 + 0j), (0.143 - 0.013j), (-0.013 - 0.079j), (-0.132 + 0.002j), (0.046 + 0.046j),
                          (0.002 - 0.132j), (-0.079 - 0.013j), (-0.013 + 0.143j), 0.092j, (-0.013 + 0.143j),
                          (-0.079 - 0.013j), (0.002 - 0.132j), (0.046 + 0.046j), (-0.132 + 0.002j), (-0.013 - 0.079j),
                          (0.143 - 0.013j), (0.092 + 0j), (0.143 - 0.013j), (-0.013 - 0.079j), (-0.132 + 0.002j),
                          (0.046 + 0.046j), (0.002 - 0.132j), (-0.079 - 0.013j), (-0.013 + 0.143j), 0.092j,
                          (-0.013 + 0.143j), (-0.079 - 0.013j), (0.002 - 0.132j), (0.046 + 0.046j), (-0.132 + 0.002j),
                          (-0.013 - 0.079j), (0.143 - 0.013j), (0.092 + 0j), (0.143 - 0.013j), (-0.013 - 0.079j),
                          (-0.132 + 0.002j), (0.046 + 0.046j), (0.002 - 0.132j), (-0.079 - 0.013j), (-0.013 + 0.143j),
                          0.092j, (-0.013 + 0.143j), (-0.079 - 0.013j), (0.002 - 0.132j), (0.046 + 0.046j),
                          (-0.132 + 0.002j), (-0.013 - 0.079j), (0.143 - 0.013j), (0.092 + 0j), (0.143 - 0.013j),
                          (-0.013 - 0.079j), (-0.132 + 0.002j), (0.046 + 0.046j), (0.002 - 0.132j), (-0.079 - 0.013j),
                          (-0.013 + 0.143j), 0.092j, (-0.013 + 0.143j), (-0.079 - 0.013j), (0.002 - 0.132j),
                          (0.046 + 0.046j), (-0.132 + 0.002j), (-0.013 - 0.079j), (0.143 - 0.013j), (0.092 + 0j),
                          (0.143 - 0.013j), (-0.013 - 0.079j), (-0.132 + 0.002j), (0.046 + 0.046j), (0.002 - 0.132j),
                          (-0.079 - 0.013j), (-0.013 + 0.143j), 0.092j, (-0.013 + 0.143j), (-0.079 - 0.013j),
                          (0.002 - 0.132j), (0.046 + 0.046j), (-0.132 + 0.002j), (-0.013 - 0.079j), (0.143 - 0.013j),
                          (0.092 + 0j), (0.143 - 0.013j), (-0.013 - 0.079j), (-0.132 + 0.002j), (0.046 + 0.046j),
                          (0.002 - 0.132j), (-0.079 - 0.013j), (-0.013 + 0.143j), 0.092j, (-0.013 + 0.143j),
                          (-0.079 - 0.013j), (0.002 - 0.132j), (0.046 + 0.046j), (-0.132 + 0.002j), (-0.013 - 0.079j),
                          (0.143 - 0.013j), (0.092 + 0j), (0.143 - 0.013j), (-0.013 - 0.079j), (-0.132 + 0.002j),
                          (0.046 + 0.046j), (0.002 - 0.132j), (-0.079 - 0.013j), (-0.013 + 0.143j), 0.092j,
                          (-0.013 + 0.143j), (-0.079 - 0.013j), (0.002 - 0.132j), (0.046 + 0.046j), (-0.132 + 0.002j),
                          (-0.013 - 0.079j), (0.143 - 0.013j), (0.092 + 0j), (0.143 - 0.013j), (-0.013 - 0.079j),
                          (-0.132 + 0.002j), (0.046 + 0.046j), (0.002 - 0.132j), (-0.079 - 0.013j), (-0.013 + 0.143j),
                          0.092j, (-0.013 + 0.143j), (-0.079 - 0.013j), (0.002 - 0.132j), (0.046 + 0.046j),
                          (-0.132 + 0.002j), (-0.013 - 0.079j), (0.143 - 0.013j), (0.092 + 0j), (0.143 - 0.013j),
                          (-0.013 - 0.079j), (-0.132 + 0.002j), (0.046 + 0.046j), (0.002 - 0.132j), (-0.079 - 0.013j),
                          (-0.013 + 0.143j), 0.092j, (-0.013 + 0.143j), (-0.079 - 0.013j), (0.002 - 0.132j)]

    # Table I-23—Time domain representation of the long training sequence
    expect_long_train = [(-0.055 + 0.023j), (0.012 - 0.098j), (0.092 - 0.106j), (-0.092 - 0.115j), (-0.003 - 0.054j),
                         (0.075 + 0.074j), (-0.127 + 0.021j), (-0.122 + 0.017j), (-0.035 + 0.151j), (-0.056 + 0.022j),
                         (-0.06 - 0.081j), (0.07 - 0.014j), (0.082 - 0.092j), (-0.131 - 0.065j), (-0.057 - 0.039j),
                         (0.037 - 0.098j), (0.062 + 0.062j), (0.119 + 0.004j), (-0.022 - 0.161j), (0.059 + 0.015j),
                         (0.024 + 0.059j), (-0.137 + 0.047j), (0.001 + 0.115j), (0.053 - 0.004j), (0.098 + 0.026j),
                         (-0.038 + 0.106j), (-0.115 + 0.055j), (0.06 + 0.088j), (0.021 - 0.028j), (0.097 - 0.083j),
                         (0.04 + 0.111j), (-0.005 + 0.12j), (0.156 + 0j), (-0.005 - 0.12j), (0.04 - 0.111j),
                         (0.097 + 0.083j), (0.021 + 0.028j), (0.06 - 0.088j), (-0.115 - 0.055j), (-0.038 - 0.106j),
                         (0.098 - 0.026j), (0.053 + 0.004j), (0.001 - 0.115j), (-0.137 - 0.047j), (0.024 - 0.059j),
                         (0.059 - 0.015j), (-0.022 + 0.161j), (0.119 - 0.004j), (0.062 - 0.062j), (0.037 + 0.098j),
                         (-0.057 + 0.039j), (-0.131 + 0.065j), (0.082 + 0.092j), (0.07 + 0.014j), (-0.06 + 0.081j),
                         (-0.056 - 0.022j), (-0.035 - 0.151j), (-0.122 - 0.017j), (-0.127 - 0.021j), (0.075 - 0.074j),
                         (-0.003 + 0.054j), (-0.092 + 0.115j), (0.092 + 0.106j), (0.012 + 0.098j), (-0.156 + 0j),
                         (0.012 - 0.098j), (0.092 - 0.106j), (-0.092 - 0.115j), (-0.003 - 0.054j), (0.075 + 0.074j),
                         (-0.127 + 0.021j), (-0.122 + 0.017j), (-0.035 + 0.151j), (-0.056 + 0.022j), (-0.06 - 0.081j),
                         (0.07 - 0.014j), (0.082 - 0.092j), (-0.131 - 0.065j), (-0.057 - 0.039j), (0.037 - 0.098j),
                         (0.062 + 0.062j), (0.119 + 0.004j), (-0.022 - 0.161j), (0.059 + 0.015j), (0.024 + 0.059j),
                         (-0.137 + 0.047j), (0.001 + 0.115j), (0.053 - 0.004j), (0.098 + 0.026j), (-0.038 + 0.106j),
                         (-0.115 + 0.055j), (0.06 + 0.088j), (0.021 - 0.028j), (0.097 - 0.083j), (0.04 + 0.111j),
                         (-0.005 + 0.12j), (0.156 + 0j), (-0.005 - 0.12j), (0.04 - 0.111j), (0.097 + 0.083j),
                         (0.021 + 0.028j), (0.06 - 0.088j), (-0.115 - 0.055j), (-0.038 - 0.106j), (0.098 - 0.026j),
                         (0.053 + 0.004j), (0.001 - 0.115j), (-0.137 - 0.047j), (0.024 - 0.059j), (0.059 - 0.015j),
                         (-0.022 + 0.161j), (0.119 - 0.004j), (0.062 - 0.062j), (0.037 + 0.098j), (-0.057 + 0.039j),
                         (-0.131 + 0.065j), (0.082 + 0.092j), (0.07 + 0.014j), (-0.06 + 0.081j), (-0.056 - 0.022j),
                         (-0.035 - 0.151j), (-0.122 - 0.017j), (-0.127 - 0.021j), (0.075 - 0.074j), (-0.003 + 0.054j),
                         (-0.092 + 0.115j), (0.092 + 0.106j), (0.012 + 0.098j), (-0.156 + 0j), (0.012 - 0.098j),
                         (0.092 - 0.106j), (-0.092 - 0.115j), (-0.003 - 0.054j), (0.075 + 0.074j), (-0.127 + 0.021j),
                         (-0.122 + 0.017j), (-0.035 + 0.151j), (-0.056 + 0.022j), (-0.06 - 0.081j), (0.07 - 0.014j),
                         (0.082 - 0.092j), (-0.131 - 0.065j), (-0.057 - 0.039j), (0.037 - 0.098j), (0.062 + 0.062j),
                         (0.119 + 0.004j), (-0.022 - 0.161j), (0.059 + 0.015j), (0.024 + 0.059j), (-0.137 + 0.047j),
                         (0.001 + 0.115j), (0.053 - 0.004j), (0.098 + 0.026j), (-0.038 + 0.106j), (-0.115 + 0.055j),
                         (0.06 + 0.088j), (0.021 - 0.028j), (0.097 - 0.083j), (0.04 + 0.111j), (-0.005 + 0.12j)]

    # Table I-24—Time domain representation of the SIGNAL field (1 symbol)
    expect_signal_field = [(0.109 + 0j), (0.033 - 0.044j), (-0.002 - 0.038j), (-0.081 + 0.084j), (0.007 - 0.1j),
                           (-0.001 - 0.113j), (-0.021 - 0.005j), (0.136 - 0.105j), (0.098 - 0.044j), (0.011 - 0.002j),
                           (-0.033 + 0.044j), (-0.06 + 0.124j), (0.01 + 0.097j), -0.008j, (0.018 - 0.083j),
                           (-0.069 + 0.027j), (-0.219 + 0j), (-0.069 - 0.027j), (0.018 + 0.083j), 0.008j,
                           (0.01 - 0.097j), (-0.06 - 0.124j), (-0.033 - 0.044j), (0.011 + 0.002j), (0.098 + 0.044j),
                           (0.136 + 0.105j), (-0.021 + 0.005j), (-0.001 + 0.113j), (0.007 + 0.1j), (-0.081 - 0.084j),
                           (-0.002 + 0.038j), (0.033 + 0.044j), (0.062 + 0j), (0.057 + 0.052j), (0.016 + 0.174j),
                           (0.035 + 0.116j), (-0.051 - 0.202j), (0.011 + 0.036j), (0.089 + 0.209j), (-0.049 - 0.008j),
                           (-0.035 + 0.044j), (0.017 - 0.059j), (0.053 - 0.017j), (0.099 + 0.1j), (0.034 - 0.148j),
                           (-0.003 - 0.094j), (-0.12 + 0.042j), (-0.136 - 0.07j), (-0.031 + 0j), (-0.136 + 0.07j),
                           (-0.12 - 0.042j), (-0.003 + 0.094j), (0.034 + 0.148j), (0.099 - 0.1j), (0.053 + 0.017j),
                           (0.017 + 0.059j), (-0.035 - 0.044j), (-0.049 + 0.008j), (0.089 - 0.209j), (0.011 - 0.036j),
                           (-0.051 + 0.202j), (0.035 - 0.116j), (0.016 - 0.174j), (0.057 - 0.052j), (0.062 + 0j),
                           (0.033 - 0.044j), (-0.002 - 0.038j), (-0.081 + 0.084j), (0.007 - 0.1j), (-0.001 - 0.113j),
                           (-0.021 - 0.005j), (0.136 - 0.105j), (0.098 - 0.044j), (0.011 - 0.002j), (-0.033 + 0.044j),
                           (-0.06 + 0.124j), (0.01 + 0.097j), -0.008j, (0.018 - 0.083j), (-0.069 + 0.027j)]

    # Table I-25—Time domain representation of the DATA field: symbol 1of 6
    expected_data1 = [(-0.139 + 0.05j), (0.004 + 0.014j), (0.011 - 0.1j), (-0.097 - 0.02j), (0.062 + 0.081j),
                      (0.124 + 0.139j), (0.104 - 0.015j), (0.173 - 0.14j), (-0.04 + 0.006j), (-0.133 + 0.009j),
                      (-0.002 - 0.043j), (-0.047 + 0.092j), (-0.109 + 0.082j), (-0.024 + 0.01j), (0.096 + 0.019j),
                      (0.019 - 0.023j), (-0.087 - 0.049j), (0.002 + 0.058j), (-0.021 + 0.228j), (-0.103 + 0.023j),
                      (-0.019 - 0.175j), (0.018 + 0.132j), (-0.071 + 0.16j), (-0.153 - 0.062j), (-0.107 + 0.028j),
                      (0.055 + 0.14j), (0.07 + 0.103j), (-0.056 + 0.025j), (-0.043 + 0.002j), (0.016 - 0.118j),
                      (0.026 - 0.071j), (0.033 + 0.177j), (0.02 - 0.021j), (0.035 - 0.088j), (-0.008 + 0.101j),
                      (-0.035 - 0.01j), (0.065 + 0.03j), (0.092 - 0.034j), (0.032 - 0.123j), (-0.018 + 0.092j), -0.006j,
                      (-0.006 - 0.056j), (-0.019 + 0.04j), (0.053 - 0.131j), (0.022 - 0.133j), (0.104 - 0.032j),
                      (0.163 - 0.045j), (-0.105 - 0.03j), (-0.11 - 0.069j), (-0.008 - 0.092j), (-0.049 - 0.043j),
                      (0.085 - 0.017j), (0.09 + 0.063j), (0.015 + 0.153j), (0.049 + 0.094j), (0.011 + 0.034j),
                      (-0.012 + 0.012j), (-0.015 - 0.017j), (-0.061 + 0.031j), (-0.07 - 0.04j), (0.011 - 0.109j),
                      (0.037 - 0.06j), (-0.003 - 0.178j), (-0.007 - 0.128j), (-0.059 + 0.1j), (0.004 + 0.014j),
                      (0.011 - 0.1j), (-0.097 - 0.02j), (0.062 + 0.081j), (0.124 + 0.139j), (0.104 - 0.015j),
                      (0.173 - 0.14j), (-0.04 + 0.006j), (-0.133 + 0.009j), (-0.002 - 0.043j), (-0.047 + 0.092j),
                      (-0.109 + 0.082j), (-0.024 + 0.01j), (0.096 + 0.019j), (0.019 - 0.023j)]

    # Table I-26—Time domain representation of the DATA field: symbol 2 of 6
    expected_data2 = [(-0.058 + 0.016j), (-0.096 - 0.045j), (-0.11 + 0.003j), (-0.07 + 0.216j), (-0.04 + 0.059j),
                      (0.01 - 0.056j), (0.034 + 0.065j), (0.117 + 0.033j), (0.078 - 0.133j), (-0.043 - 0.146j),
                      (0.158 - 0.071j), (0.254 - 0.021j), (0.068 + 0.117j), (-0.044 + 0.114j), (-0.035 + 0.041j),
                      (0.085 + 0.07j), (0.12 + 0.01j), (0.057 + 0.055j), (0.063 + 0.188j), (0.091 + 0.149j),
                      (-0.017 - 0.039j), (-0.078 - 0.075j), (0.049 + 0.079j), (-0.014 - 0.007j), (0.03 - 0.027j),
                      (0.08 + 0.054j), (-0.186 - 0.067j), (-0.039 - 0.027j), (0.043 - 0.072j), (-0.092 - 0.089j),
                      (0.029 + 0.105j), (-0.144 + 0.003j), (-0.069 - 0.041j), (0.132 + 0.057j), (-0.126 + 0.07j),
                      (-0.031 + 0.109j), (0.161 - 0.009j), (0.056 - 0.046j), (-0.004 + 0.028j), (-0.049 + 0j),
                      (-0.078 - 0.005j), (0.015 - 0.087j), (0.149 - 0.104j), (-0.021 - 0.051j), (-0.154 - 0.106j),
                      (0.024 + 0.03j), (0.046 + 0.123j), (-0.004 - 0.098j), (-0.061 - 0.128j), (-0.024 - 0.038j),
                      (0.066 - 0.048j), (-0.067 + 0.027j), (0.054 - 0.05j), (0.171 - 0.049j), (-0.108 + 0.132j),
                      (-0.161 - 0.019j), (-0.07 - 0.072j), (-0.177 + 0.049j), (-0.172 - 0.05j), (0.051 - 0.075j),
                      (0.122 - 0.057j), (0.009 - 0.044j), (-0.012 - 0.021j), (0.004 + 0.009j), (-0.03 + 0.081j),
                      (-0.096 - 0.045j), (-0.11 + 0.003j), (-0.07 + 0.216j), (-0.04 + 0.059j), (0.01 - 0.056j),
                      (0.034 + 0.065j), (0.117 + 0.033j), (0.078 - 0.133j), (-0.043 - 0.146j), (0.158 - 0.071j),
                      (0.254 - 0.021j), (0.068 + 0.117j), (-0.044 + 0.114j), (-0.035 + 0.041j), (0.085 + 0.07j)]

    # Table I-27—Time domain representation of the DATA field: symbol 3 of 6
    expected_data3 = [(0.001 + 0.011j), (-0.099 - 0.048j), (0.054 - 0.196j), (0.124 + 0.035j), (0.092 + 0.045j),
                      (-0.037 - 0.066j), (-0.021 - 0.004j), (0.042 - 0.065j), (0.061 + 0.048j), (0.046 + 0.004j),
                      (-0.063 - 0.045j), (-0.102 + 0.152j), (-0.039 - 0.019j), (-0.005 - 0.106j), (0.083 + 0.031j),
                      (0.226 + 0.028j), (0.14 - 0.01j), (-0.132 - 0.033j), (-0.116 + 0.088j), (0.023 + 0.052j),
                      (-0.171 - 0.08j), (-0.246 - 0.025j), (-0.062 - 0.038j), (-0.055 - 0.062j), (-0.004 - 0.06j),
                      (0.034 + 0j), (-0.03 + 0.021j), (0.075 - 0.122j), (0.043 - 0.08j), (-0.022 + 0.041j),
                      (0.026 + 0.013j), (-0.031 - 0.018j), (0.059 + 0.008j), (0.109 + 0.078j), (0.002 + 0.101j),
                      (-0.016 + 0.054j), (-0.059 + 0.07j), (0.017 + 0.114j), (0.104 - 0.034j), (-0.024 - 0.059j),
                      (-0.081 + 0.051j), (-0.04 - 0.069j), (-0.069 + 0.058j), (-0.067 + 0.117j), (0.007 - 0.131j),
                      (0.009 + 0.028j), (0.075 + 0.117j), (0.118 + 0.03j), (-0.041 + 0.148j), (0.005 + 0.098j),
                      (0.026 + 0.002j), (-0.116 + 0.045j), (-0.02 + 0.084j), (0.101 + 0.006j), (0.205 - 0.064j),
                      (0.073 - 0.063j), (-0.174 - 0.118j), (-0.024 + 0.026j), (-0.041 + 0.129j), (-0.042 - 0.053j),
                      (0.148 - 0.126j), (-0.03 - 0.049j), (-0.015 - 0.021j), (0.089 - 0.069j), (-0.119 + 0.011j),
                      (-0.099 - 0.048j), (0.054 - 0.196j), (0.124 + 0.035j), (0.092 + 0.045j), (-0.037 - 0.066j),
                      (-0.021 - 0.004j), (0.042 - 0.065j), (0.061 + 0.048j), (0.046 + 0.004j), (-0.063 - 0.045j),
                      (-0.102 + 0.152j), (-0.039 - 0.019j), (-0.005 - 0.106j), (0.083 + 0.031j), (0.226 + 0.028j)]

    # Table I-28—Time domain representation of the DATA field: symbol 4 of 6
    expected_data4 = [(0.085 - 0.065j), (0.034 - 0.142j), (0.004 - 0.012j), (0.126 - 0.043j), (0.055 + 0.068j),
                      (-0.02 + 0.077j), (0.008 - 0.056j), (-0.034 + 0.046j), (-0.04 - 0.134j), (-0.056 - 0.131j),
                      (0.014 + 0.097j), (0.045 - 0.009j), (-0.113 - 0.17j), (-0.065 - 0.23j), (0.065 - 0.011j),
                      (0.011 + 0.048j), (-0.091 - 0.059j), (-0.11 + 0.024j), (0.074 - 0.034j), (0.124 + 0.022j),
                      (-0.037 + 0.071j), (0.015 + 0.002j), (0.028 + 0.099j), (-0.062 + 0.068j), (0.064 + 0.016j),
                      (0.078 + 0.156j), (0.009 + 0.219j), (0.147 + 0.024j), (0.106 + 0.03j), (-0.08 + 0.143j),
                      (-0.049 - 0.1j), (-0.036 - 0.082j), (-0.089 + 0.021j), (-0.07 - 0.029j), (-0.086 + 0.048j),
                      (-0.066 - 0.015j), (-0.024 + 0.002j), (-0.03 - 0.023j), (-0.032 + 0.02j), (-0.002 + 0.212j),
                      (0.158 - 0.024j), (0.141 - 0.119j), (-0.146 + 0.058j), (-0.155 + 0.083j), (-0.002 - 0.03j),
                      (0.018 - 0.129j), (0.012 - 0.018j), (-0.008 - 0.037j), (0.031 + 0.04j), (0.023 + 0.097j),
                      (0.014 - 0.039j), (0.05 + 0.019j), (-0.072 - 0.141j), (-0.023 - 0.051j), (0.024 + 0.099j),
                      (-0.127 - 0.116j), (0.094 + 0.102j), (0.183 + 0.098j), (-0.04 - 0.02j), (0.065 + 0.077j),
                      (0.088 - 0.147j), (-0.039 - 0.059j), (-0.057 + 0.124j), (-0.077 + 0.02j), (0.03 - 0.12j),
                      (0.034 - 0.142j), (0.004 - 0.012j), (0.126 - 0.043j), (0.055 + 0.068j), (-0.02 + 0.077j),
                      (0.008 - 0.056j), (-0.034 + 0.046j), (-0.04 - 0.134j), (-0.056 - 0.131j), (0.014 + 0.097j),
                      (0.045 - 0.009j), (-0.113 - 0.17j), (-0.065 - 0.23j), (0.065 - 0.011j), (0.011 + 0.048j),
                      (-0.026 - 0.021j), (-0.002 + 0.041j), (0.001 + 0.071j), (-0.037 - 0.117j)]

    # Table I-29—Time domain representation of the DATA field: symbol 5 of 6
    expected_data5 = [(-0.106 - 0.062j), (0.002 + 0.057j), (-0.008 - 0.011j), (0.019 + 0.072j), (0.016 + 0.059j),
                      (-0.065 - 0.077j), (0.142 - 0.062j), (0.087 + 0.025j), (-0.003 - 0.103j), (0.107 - 0.152j),
                      (-0.054 + 0.036j), (-0.03 - 0.003j), (0.058 - 0.02j), (-0.028 + 0.007j), (-0.027 - 0.099j),
                      (0.049 - 0.075j), (0.174 + 0.031j), (0.134 + 0.156j), (0.06 + 0.077j), (-0.01 - 0.022j),
                      (-0.084 + 0.04j), (-0.074 + 0.011j), (-0.163 + 0.054j), (-0.052 - 0.008j), (0.076 - 0.042j),
                      (0.043 + 0.101j), (0.058 - 0.018j), (0.003 - 0.09j), (0.059 - 0.018j), (0.023 - 0.031j),
                      (0.007 - 0.017j), (0.066 - 0.017j), (-0.135 - 0.098j), (-0.056 - 0.081j), (0.089 + 0.154j),
                      (0.12 + 0.122j), (0.102 + 0.001j), (-0.141 + 0.102j), (0.006 - 0.011j), (0.057 - 0.039j),
                      (-0.059 + 0.066j), (0.132 + 0.111j), (0.012 + 0.114j), (0.047 - 0.106j), (0.16 - 0.099j),
                      (-0.076 + 0.084j), (-0.049 + 0.073j), (0.005 - 0.086j), (-0.052 - 0.108j), (-0.073 + 0.129j),
                      (-0.129 - 0.034j), (-0.153 - 0.111j), (-0.193 + 0.098j), (-0.107 - 0.068j), (0.004 - 0.009j),
                      (-0.039 + 0.024j), (-0.054 - 0.079j), (0.024 + 0.084j), (0.052 - 0.002j), (0.028 - 0.044j),
                      (0.04 + 0.018j), (-0.002 + 0.041j), (0.001 + 0.071j), (-0.037 - 0.117j), (-0.106 - 0.062j),
                      (0.002 + 0.057j), (-0.008 - 0.011j), (0.019 + 0.072j), (0.016 + 0.059j), (-0.065 - 0.077j),
                      (0.142 - 0.062j), (0.087 + 0.025j), (-0.003 - 0.103j), (0.107 - 0.152j), (-0.054 + 0.036j),
                      (-0.03 - 0.003j)]

    # Table I-30—Time domain representation of the DATA field: symbol 6 of 6
    expected_data6 = [(0.029 - 0.026j), (-0.047 + 0.077j), (-0.007 - 0.002j), (0.05 - 0.021j), (0.046 - 0.04j),
                      (-0.061 - 0.099j), (-0.121 + 0.008j), (0.014 + 0.05j), (0.145 + 0.034j), (0.001 - 0.046j),
                      (-0.058 - 0.121j), (0.04 + 0.001j), (-0.029 + 0.041j), (0.002 - 0.066j), (0.015 - 0.054j),
                      (0.01 - 0.029j), (0.008 - 0.119j), (-0.134 + 0.002j), (0.064 + 0.079j), (0.095 - 0.102j),
                      (-0.069 - 0.014j), (0.156 + 0.037j), (0.047 - 0.008j), (-0.076 + 0.025j), (0.117 - 0.143j),
                      (0.056 - 0.042j), (0.002 + 0.075j), (-0.039 - 0.058j), (-0.092 + 0.014j), (-0.041 + 0.047j),
                      (-0.058 + 0.092j), (0.012 + 0.154j), (0.079 + 0.091j), (-0.067 + 0.017j), (-0.102 - 0.032j),
                      (0.039 + 0.084j), (-0.036 + 0.014j), (-0.001 - 0.046j), (0.195 + 0.131j), (0.039 + 0.067j),
                      (-0.007 + 0.045j), (0.051 + 0.008j), (-0.074 - 0.109j), (-0.033 + 0.07j), (-0.028 + 0.176j),
                      (-0.041 + 0.045j), (0.014 - 0.084j), (0.054 - 0.04j), (0.11 - 0.02j), (0.014 - 0.021j),
                      (0.006 + 0.139j), (0.008 + 0.011j), (-0.06 - 0.04j), (0.008 + 0.179j), (0.008 + 0.02j),
                      (0.044 - 0.114j), (0.021 - 0.015j), (-0.008 - 0.052j), (0.091 - 0.109j), (-0.025 - 0.04j),
                      (-0.049 + 0.006j), (-0.043 - 0.041j), (-0.178 - 0.026j), (-0.073 - 0.057j), -0.031j,
                      (-0.047 + 0.077j), (-0.007 - 0.002j), (0.05 - 0.021j), (0.046 - 0.04j), (-0.061 - 0.099j),
                      (-0.121 + 0.008j), (0.014 + 0.05j), (0.145 + 0.034j), (0.001 - 0.046j), (-0.058 - 0.121j),
                      (0.04 + 0.001j), (-0.029 + 0.041j), (0.002 - 0.066j), (0.015 - 0.054j), (0.01 - 0.029j),
                      (0.004 - 0.059j)]

    expected = np.concatenate(
        [expect_short_train, expect_long_train, expect_signal_field, expected_data1, expected_data2, expected_data3,
         expected_data4, expected_data5, expected_data6])

    output = np.round(transmitter(input, data_rate=36), 3)
    np.testing.assert_equal(output, expected)
