"""
d) Append the PSDU to the SERVICE field of the TXVECTOR. Extend the resulting bit string with
zero bits (at least 6 bits) so that the resulting length is a multiple of N DBPS . The resulting bit string
constitutes the DATA part of the packet. Refer to 17.3.5.4 for details.

1. Appends the SERVICE field
2. Extends with 0 bits, so that enough bits for OFDM symbol

The SERVICE field has 16 bits. The bits from 0–6 of the SERVICE field, which are transmitted first, are set to 0s and are used to
synchronize the descrambler in the receiver. The remaining 9 bits (7–15) of the SERVICE field shall be
reserved for future use. All reserved bits shall be set to 0 on transmission and ignored on reception. Refer to
Figure 17-6.

The PPDU TAIL field shall be six bits of 0, which are required to return the convolutional encoder to the
zero state. This procedure improves the error probability of the convolutional decoder, which relies on future
bits when decoding and which may be not be available past the end of the message.
"""
from typing import Tuple

from hypothesis import given
from hypothesis._strategies import binary, sampled_from

from wifi import bits, bitstr
import numpy as np


def do(data: bits, data_bits_per_ofdm_symbol: int) -> Tuple[bits, int]:
    service = '0' * 16
    tail = '0' * 6
    data = service + data + tail

    n_symbols = int(np.ceil(len(data) / data_bits_per_ofdm_symbol))
    n_data = n_symbols * data_bits_per_ofdm_symbol
    n_pad = int(n_data - len(data))
    pad = '0' * n_pad

    data = data + pad
    return data, n_pad


def undo(data: bits, length_bytes: int) -> bits:
    return data[16:16 + length_bytes * 8]


@given(binary(), sampled_from([48, 96, 192, 288]))
def test_hypothesis(data, data_bits_per_ofdm_symbol):
    data = bitstr.from_bytes(data)

    done_data, n_pad = do(data, data_bits_per_ofdm_symbol)
    assert undo(done_data, len(data) // 8) == data
