import numpy as np
from wifi.util import flip_byte_endian


class bits(str):

    @classmethod
    def from_int(cls, x: int, bits: int):
        return cls(bin(x)[2:].zfill(bits))  # [2:] skips the '0b' string

    def __init__(self, val):
        if isinstance(val, list):
            val = ''.join(val)
        elif isinstance(val, np.ndarray):
            val = ''.join([str(int(x)) for x in val])
        elif val[0:2] in ('0x', '0X'):
                val = val[2:]
                num_of_bits = int(len(val) * np.log2(16))

                val = cls.from_int(int(val, 16), num_of_bits)
                val = flip_byte_endian(val)  # IEE802.11 examples need this?!, tho it is confusing

        self.data = val

    def __str__(self):
        return self.data.__str__()

    def __repr__(self):
        return self.data.__repr__()

    def __eq__(self, other):
        return self.data == other

    def __hash__(self):
        return self.data.__hash__()

    def __iter__(self):
        return self.data.__iter__()

    def __getitem__(self, item) -> 'bits':
        if isinstance(item, (tuple, list)):
            # use Numpy fancy indexing. Example: a[1, 2] can select element 1 and 2
            num = np.array([x for x in self])
            res = num[list(item)].tolist()
        else:
            res = self.data[item]
        return bits(res)

    def __add__(self, other):
        """
        Adds support for appending int and bool items.
        # >>> bits('010') + 1
        # '0101'
        #
        # >>> bits('010') + False
        # '0100'
        """
        if isinstance(other, int):  # int also covers bool
            assert other == 0 or other == 1
            other = str(int(other))

        self.data += other
        return self

    # def __xor__(self, other):


def test_from_int():
    assert bits.from_int(1, 2) == '01'


def test_append():
    a = bits('0011')
    res = a + '0'
    assert res == '00110'
    assert type(res) == bits


def test_append_int():
    a = bits('0011')
    res = a + 1
    assert res == '00111'
    assert type(res) == bits


def test_append_bool():
    a = bits('0011')
    res = a + True
    assert res == '00111'
    assert type(res) == bits


class TestInit:
    def test_list(self):
        assert bits(['0', '1', '0']) == '010'

    def test_ndarray(self):
        assert bits(np.array([0, 1, 0])) == '010'


def test_getitem():
    a = bits('1234')
    res = a[0]
    assert res == '1'
    assert type(res) == bits


def test_getitem_fancy():
    a = bits('1234')
    res = a[0, 1]
    assert res == '12'
    assert type(res) == bits  # need this to support bit operations like XOR
