"""
Puncturing is a procedure for omitting some of the encoded bits in the transmitter
(thus reducing the number of transmitted bits and increasing the coding rate)
and inserting a dummy “zero” metric into the convolutional decoder on the
receive side in place of the omitted bits.
The puncturing patterns are illustrated in Figure 17-9.
"""
from hypothesis import assume, given
from hypothesis._strategies import binary, sampled_from

from wifi import bits
from wifi.util import is_divisible


def do(data: bits, coding_rate='1/2') -> bits:

    if coding_rate == '2/3':
        # throw out each 3. bit from groups of 4 bits
        assume(is_divisible(data, by=4))
        data = [bit for i, bit in enumerate(data) if (i % 4) != 3]
    elif coding_rate == '3/4':
        # throw out each 3. and 4. bit groups of 6 bits
        assume(is_divisible(data, by=6))
        data = [bit for i, bit in enumerate(data) if (i % 6) != 3 and (i % 6) != 4]

    return bits(data)


def undo(data: bits, coding_rate='1/2') -> bits:
    # un-puncturing process i.e. add 'X' bits, which are basically just ignored by the conv decoder
    if coding_rate == '3/4':
        assume(is_divisible(data, by=4))
        data = [d[:3] + '??' + d[3] for d in data.split(4)]
    elif coding_rate == '2/3':
        assume(is_divisible(data, by=3))
        data = [d + '?' for d in data.split(3)]
    return bits(data)


@given(binary(), sampled_from(['1/2', '2/3', '3/4']))
def test_hypothesis(data, coding_rate):
    data = bits(data)

    # cant test equality because 'do' throws away data
    do1 = do(data, coding_rate)
    undo1 = undo(do1, coding_rate)
    assert len(undo1) == len(data)

    do2 = do(undo1, coding_rate)
    assert do1 == do2
