from typing import List
import numpy as np
from hypothesis import given, assume
from hypothesis._strategies import integers, binary
from more_itertools import chunked

bits = str


def from_int(i, number_of_bits=None) -> bits:
    assume(i >= 0)
    b = bin(i)[2:]
    if number_of_bits:
        b = b.zfill(number_of_bits)
    return b


def to_int(bits) -> int:
    assume(bits != '')
    return int(bits, 2)


def from_hex(hexstr) -> bits:
    assume(hexstr[0] != '-')  # negative hex??
    val = hexstr[2:]
    if not val:
        return ''
    num_of_bits = int(len(val) * np.log2(16))
    val = bin(int(val, 16))[2:].zfill(num_of_bits)
    return val


def to_hex(bits) -> str:
    return hex(to_int(bits))


def from_bytes(bytes) -> bits:
    val = '0x' + bytes.hex()
    return from_hex(val)


def to_bytes(bits) -> bytes:
    return bytes([to_int(''.join(x)) for x in chunked(bits, 8)])


def split(bits, split_size) -> List[bits]:
    assume(is_divisible(bits, split_size))
    return [bits[i:i + split_size] for i in range(0, len(bits), split_size)]


def merge(bits: List[bits]):
    return ''.join(bits)


def from_list(l):
    r = [str(int(x)) for x in l]
    return merge(r)


def reverse(x: bits) -> bits:
    return x[::-1]


def is_divisible(x:bits, by:int) -> bool:
    return len(x) != 0 and len(x) % by == 0


@given(integers())
def test_int(i):
    assert to_int(from_int(i)) == i


@given(integers())
def test_hex(i):
    i = hex(i)
    assert to_hex(from_hex(i)) == i


@given(binary())
def test_bytes(i):
    assert to_bytes(from_bytes(i)) == i


def test_fill():
    assert from_int(0, 3) == '000'


def test_split_merge():
    b = from_hex('0x12')
    s = split(b, 4)
    assert s == ['0001', '0010']

    m = merge(s)
    assert m == b


def test_list():
    assert from_list([1, 0, 0, 1]) == '1001'
