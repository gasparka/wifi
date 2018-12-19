from typing import Tuple

from wifi.bits import bits
from wifi.util import reverse

"""
Data rate to bits:
| R1–R4  | Rate (Mb/s) (20 MHz channel) |
|--------|------------------------------|
| 1101   | 6                            |
| 1111   | 9                            |
| 0101   | 12                           |
| 0111   | 18                           |
| 1001   | 24                           |
| 1011   | 36                           |
| 0001   | 48                           |
| 0011   | 54                           |
"""
RATE_LUT = {6: '1101',
            9: '1111',
            12: '0101',
            18: '0111',
            24: '1001',
            36: '1011',
            48: '0001',
            54: '0011'}


def encode(data_rate: int, length_bytes: int) -> bits:
    """
    17.3.4 SIGNAL field - IEEE Std 802.11-2016

    The OFDM training symbols shall be followed by the SIGNAL field, which contains the RATE and the
    LENGTH fields of the TXVECTOR. The RATE field conveys information about the type of modulation
    and the coding rate as used in the rest of the packet.

    The SIGNAL field is composed of 24 bits:
    RATE        RESERVED    LENGTH      PARITY  SIGNAL_TAIL
    (4 bits)    (1 bit)     (12 bits)   (1 bit) (6 bits)

    """
    if length_bytes > (2 ** 12) - 1:
        raise Exception(f'Maximum bytes in a packet is {(2**12)-1}, you require {length_bytes}')

    # First 4 bits indicate rate
    signal = bits(RATE_LUT[data_rate])

    # Bit 4 is reserved. It shall be set to 0 on transmit and ignored on receive.
    signal += '0'

    # Data length
    signal += reverse(bits.from_int(length_bytes, bits=12))

    # Bit 17 shall be a positive parity (even-parity) bit for bits 0–16
    signal += signal.count('1') & 1

    # In order to facilitate a reliable and timely detection of the RATE and LENGTH fields, 6 zero tail bits are
    # inserted into the PHY header.
    signal += '000000'
    return signal


def decode(data: bits) -> Tuple[int, int]:
    parity = data[:17].count('1') & 1
    assert parity == data[17]

    data_rate_bits = data[:4]
    data_rate = [key for key, value in RATE_LUT.items() if value == data_rate_bits][0]
    length_bytes = int(reverse(data[5:17]), 2)
    return data_rate, length_bytes


def test_signal_field():
    """
    IEEE Std 802.11-2016: I.1.4.1 SIGNAL field bit assignment
    """

    # IEEE Std 802.11-2016: Table I-7—Bit assignment for SIGNAL field
    expect = '101100010011000000000000'
    data_rate = 36
    length_bytes = 100
    output = encode(data_rate, length_bytes)
    assert output == expect

    # test decode
    dec_data_rate, dec_length_bytes = decode(output)
    assert dec_data_rate == data_rate
    assert dec_length_bytes == length_bytes
