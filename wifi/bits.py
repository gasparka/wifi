from typing import List, Optional

import numpy as np
from wifi.util import flip_byte_endian


class bits:

    @classmethod
    def from_int(cls, x: int, bits: int):
        """

        >>> bits.from_int(1, 2)
        '01'
        """
        return cls(bin(x)[2:].zfill(bits))  # [2:] skips the '0b' string

    def __init_subclass__(cls, **kwargs):
        super().__init_subclass__(**kwargs)
        cls.subclasses.append(str)

    def __init__(self, val):
        """
        >>> bits(['0', '1', '0'])
        '010'

        >>> bits(np.array([0, 1, 0]))
        '010'

        >>> bits(['01', '10'])
        '0110'
        """
        if isinstance(val, list):
            val = ''.join(val)
        elif isinstance(val, np.ndarray):
            val = ''.join([str(int(x)) for x in val])
        elif isinstance(val, str) and val[0:2] in ('0x', '0X'):
            val = val[2:]
            num_of_bits = int(len(val) * np.log2(16))

            val = bits.from_int(int(val, 16), num_of_bits)
            val = flip_byte_endian(val)  # IEE802.11 examples need this?!, tho it is confusing

        self.data = val

    def __str__(self):
        return self.data.__str__()

    def __repr__(self):
        return self.data.__repr__()

    def __eq__(self, other):
        """
        >>> bits('0') == 0
        True
        """
        if isinstance(other, int):  # int also covers bool
            assert other == 0 or other == 1
            other = str(int(other))
        return self.data == other

    def __hash__(self):
        return self.data.__hash__()

    def __iter__(self):
        return self.data.__iter__()

    def __getitem__(self, item) -> 'bits':
        """
        >>> a = bits('01001')
        >>> a[0]
        '0'
        >>> type(a[0])
        <class 'bits.bits'>
        >>> a[0, 2]
        '00'
        """
        if isinstance(item, (tuple, list)):
            # use Numpy fancy indexing. Example: a[1, 2] can select element 1 and 2
            num = np.array([x for x in self.data])
            res = num[list(item)].tolist()
        else:
            res = self.data[item]
        return bits(res)

    def __add__(self, other):
        """
        >>> bits('0011') + '0'
        '00110'

        Adds support for appending int and bool items.
        >>> bits('010') + 1
        '0101'

        >>> bits('010') + False
        '0100'
        """
        if isinstance(other, int):  # int also covers bool
            assert other == 0 or other == 1
            other = str(int(other))

        self.data += str(other)
        return self

    def __radd__(self, other):
        """
        >>> '0' + bits('0011')
        '00011'

        Adds support for appending int and bool items.
        >>> 1 + bits('010')
        '1010'

        >>> False + bits('010')
        '0010'
        """
        if isinstance(other, int):  # int also covers bool
            assert other == 0 or other == 1
            other = str(int(other))

        self.data = str(other) + self.data
        return self

    def reshape(self, shape) -> List['bits']:
        """
        >>> bits('0011').reshape((-1, 2))
        ['00', '11']
        """
        assert shape[0] == -1 # just to look similar to numpy TODO
        return [bits(self.data[i:i + shape[1]]) for i in range(0, len(self.data), shape[1])]

    def astype(self, type):
        """
        >>> bits('0011').astype(int)
        3
        """
        if type == int:
            return int(self.data, 2)
        assert False

    def count(self, x):
        """
        >>> bits('1111011').count('1')
        6
        """
        return self.data.count(x)

    def __len__(self):
        return len(self.data)

    # def __xor__(self, other):

def test_wtf():
    a = bits('123')
    pass