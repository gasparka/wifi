import numpy as np


class bits(str):
    def __new__(cls, val):
        # if val[0:2] in ('0x', '0X'):
        #     val = val[2:]
        #     num_of_bits = int(len(val) * np.log2(16))
        #     val = int_to_binstr(int(val, 16), num_of_bits)
        #     val = flip_byte_endian(val)  # IEE802.11 examples need this?!, tho it is confusing

        if isinstance(val, list):
            val = ''.join(val)

        r = str.__new__(cls, val)
        return r

    def __getitem__(self, item) -> 'bits':
        if isinstance(item, (tuple, list)):
            # use Numpy fancy indexing. Example: a[1, 2] can select element 1 and 2
            num = np.array([x for x in self])
            res = num[list(item)].tolist()
        else:
            res = super().__getitem__(item)
        return bits(res)

    # def __xor__(self, other):


class TestInit:
    def test_list(self):
        assert bits(['0', '1', '0']) == '010'


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
