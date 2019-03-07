import numpy as np
from hypothesis import given, assume
from hypothesis._strategies import integers, binary
from more_itertools import chunked

bits = str  # alias for better annotations


def from_int(i, number_of_bits=None) -> bits:
    assume(i >= 0)
    b = bin(i)[2:]
    if number_of_bits:
        b.zfill(number_of_bits)
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


@given(integers())
def test_int(i):
    assert to_int(from_int(i)) == i


@given(integers())
def test_hex(i):
    i = hex(i)
    assert to_hex(from_hex(i)) == i


@given(binary())
def test_bytes(i):
    f = from_bytes(i)
    b = to_bytes(f)
    assert to_bytes(from_bytes(i)) == i

# from typing import List
# import numpy as np
#
#
# class bits:
#
#     @classmethod
#     def from_int(cls, x: int, bits: int):
#         """
#
#         >>> bits.from_int(1, 2)
#         '01'
#         """
#         return cls(bin(x)[2:].zfill(bits))  # [2:] skips the '0b' string
#
#     def __init_subclass__(cls, **kwargs):
#         super().__init_subclass__(**kwargs)
#         cls.subclasses.append(str)
#
#     def __init__(self, val, init_only=False):
#         """
#         >>> bits(['0', '1', '0'])
#         '010'
#
#         >>> bits(np.array([0, 1, 0]))
#         '010'
#
#         >>> bits(['01', '10'])
#         '0110'
#
#         >>> bits([bits('01'), bits('10')])
#         '0110'
#
#         >>> bits('0x1234')
#         '0001001000110100'
#
#         >>> bits(b'')
#         ''
#
#         >>> bits(b'tere')
#         '01110100011001010111001001100101'
#
#         """
#         if init_only:
#             self.data=val
#             return
#
#         if isinstance(val, bytes):
#             val = val.hex()
#             if val != '':
#                 val = '0x' + val
#
#         if isinstance(val, list):
#             import operator, functools
#             val = functools.reduce(operator.add, val)
#         elif isinstance(val, np.ndarray):
#             val = ''.join([str(int(x)) for x in val]) # used in interleaver
#         elif isinstance(val, str) and val[0:2] in ('0x', '0X'):
#             val = val[2:]
#             num_of_bits = int(len(val) * np.log2(16))
#             val = bits.from_int(int(val, 16), num_of_bits)
#             # val = flip_byte_endian(val)  # IEE802.11 examples need this?!, tho it is confusing
#
#         self.data = val
#
#     def __str__(self):
#         return self.data.__str__()
#
#     def __repr__(self):
#         return self.data.__repr__()
#
#     def __eq__(self, other):
#         """
#         >>> bits('0') == 0
#         True
#         """
#         if isinstance(other, int):  # int also covers bool
#             assert other == 0 or other == 1
#             other = str(int(other))
#         return self.data == other
#
#     def __hash__(self):
#         return self.data.__hash__()
#
#     def __iter__(self):
#         return self.data.__iter__()
#
#     def __getitem__(self, item) -> 'bits':
#         """
#         >>> a = bits('01001')
#         >>> a[0]
#         '0'
#         >>> a[0, 2]
#         '00'
#         """
#         try:
#             res = self.data[item]
#             return bits(res, init_only=True)
#         except:
#             # use Numpy fancy indexing. Example: a[1, 2] can select element 1 and 2
#             num = np.array([x for x in self.data])
#             res = num[list(item)].tolist()
#             return bits(res)
#
#     def __add__(self, other):
#         """
#         >>> bits('0011') + '0'
#         '00110'
#
#         Adds support for appending int and bool items.
#         >>> bits('010') + 1
#         '0101'
#
#         >>> bits('010') + False
#         '0100'
#
#         Must not mutate the current object!
#         >>> a = bits('0')
#         >>> a + '1'
#         '01'
#         >>> a
#         '0'
#         """
#         if isinstance(other, int):
#             other = '1' if other else '0'
#         return bits(f'{self.data}{other}', init_only=True)
#
#     def __radd__(self, other):
#         """
#         >>> '0' + bits('0011')
#         '00011'
#
#         Adds support for appending int and bool items.
#         >>> 1 + bits('010')
#         '1010'
#
#         >>> False + bits('010')
#         '0010'
#
#         Must not mutate the current object!
#         >>> a = bits('0')
#         >>> '1' + a
#         '10'
#         >>> a
#         '0'
#         """
#         if isinstance(other, int):
#             other = '1' if other else '0'
#
#         return bits(f'{other}{self.data}', init_only=True)
#
#     def astype(self, type):
#         """
#         >>> bits('0011').astype(int)
#         3
#         """
#         if type == int:
#             return int(str(self.data), 2)
#         assert False
#
#     def count(self, x):
#         """
#         >>> bits('1111011').count('1')
#         6
#         """
#         return self.data.count(x)
#
#     def flip(self):
#         """
#         >>> bits('01001').flip()
#         '10010'
#         """
#         return bits(self.data[::-1])
#
#     def split(self, amount: int) -> List['bits']:
#         """
#         >>> bits('0011').split(2)
#         ['00', '11']
#         """
#         return [bits(self.data[i:i + amount]) for i in range(0, len(self.data), amount)]
#
#     def __len__(self):
#         return len(self.data)
#
#     def __xor__(self, other):
#         """
#         >>> bits('1') ^ bits('1')
#         '0'
#
#         >>> bits('111') ^ bits('11')
#         '100'
#         """
#         res = self.astype(int) ^ other.astype(int)
#         return bits.from_int(res, bits=max(len(self), len(other)))
#
#     def __rxor__(self, other):
#         """
#         >>> '1' ^ bits('1')
#         '0'
#         """
#         return bits(other) ^ self
#
#     def __bytes__(self):
#         """
#         >>> bytes(bits(b'test'))
#         b'test'
#         """
#         ints = [int(str(x), 2) for x in self.split(8)]
#         return bytes(ints)
