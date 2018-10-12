"""
See 17.3.4 SIGNAL field - IEEE Std 802.11-2016

The OFDM training symbols shall be followed by the SIGNAL field, which contains the RATE and the
LENGTH fields of the TXVECTOR. The RATE field conveys information about the type of modulation
and the coding rate as used in the rest of the packet. The encoding of the SIGNAL single OFDM symbol
shall be performed with BPSK modulation of the subcarriers and using convolutional coding at R = 1/2. The
encoding procedure, which includes convolutional encoding, interleaving, modulation mapping processes,
pilot insertion, and OFDM modulation, follows the steps described in 17.3.5.6, 17.3.5.7, and 17.3.5.9, as
used for transmission of data with BPSK-OFDM modulated at coding rate 1/2. The contents of the SIGNAL
field are not scrambled.

"""

rate_lut_20m = {6: '1101',
                9: '1111',
                12: '0101',
                18: '0111',
                24: '1001',
                36: '1011',
                48: '0001',
                54: '0011'}


def signal_field(data_rate, length_bytes, channel='20M'):
    """
    The SIGNAL field is composed of 24 bits:

    RATE        RESERVED    LENGTH      PARITY  SIGNAL_TAIL
    (4 bits)    (1 bit)     (12 bits)   (1 bit) (6 bits)

    """
    assert channel == '20M'
    signal = ''

    """
    The bits R1–R4 shall be set, dependent on datarate:
    | R1–R4  | Rate (Mb/s) (20 MHz channel) | Rate (Mb/s) (10 MHz channel) | Rate (Mb/s) (5 MHz channel) |
    |--------|------------------------------|------------------------------|-----------------------------|
    | 1101   | 6                            | 3                            | 1.5                         |
    | 1111   | 9                            | 4.5                          | 2.25                        |
    | 0101   | 12                           | 6                            | 3                           |
    | 0111   | 18                           | 9                            | 4.5                         |
    | 1001   | 24                           | 12                           | 6                           |
    | 1011   | 36                           | 18                           | 9                           |
    | 0001   | 48                           | 24                           | 12                          |
    | 0011   | 54                           | 27                           | 13.5                        |
    """
    rate_bits = rate_lut_20m[data_rate]
    signal += rate_bits

    """ Bit 4 is reserved. It shall be set to 0 on transmit and ignored on receive. """
    signal += '0'

    """ The PHY LENGTH field shall be an unsigned 12-bit integer that indicates the number of octets in the PSDU
    that the MAC is currently requesting the PHY to transmit."""
    length_octets_bin = bin(length_bytes)[2:].zfill(12)[::-1]  # [::-1] stuff is to reverse the string
    signal += length_octets_bin

    """ Bit 17 shall be a positive parity (evenparity) bit for bits 0–16 """
    parity = signal.count('1') & 1
    signal += str(parity)

    """ 
    In order to facilitate a reliable and timely
    detection of the RATE and LENGTH fields, 6 zero tail bits are inserted into the PHY header. 
    The bits 18–23 constitute the SIGNAL TAIL field, and all 6 bits shall be set to 0. """
    signal += '000000'
    return signal


def test_signal_field():
    """
    IEEE Std 802.11-2016: I.1.4.1 SIGNAL field bit assignment
    """

    # IEEE Std 802.11-2016: Table I-7—Bit assignment for SIGNAL field
    expect = '101100010011000000000000'
    output = signal_field(36, 100)
    assert output == expect
