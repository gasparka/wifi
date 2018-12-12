# from ieee80211phy.receiver import packet_detector, receiver
from ieee80211phy.util import bit_error_rate
import numpy as np
# r = bit_error_rate(25)

#
# @profile
# def l():
#     iq = np.load('/home/gaspar/git/ieee80211phy/data/limemini_air.npy')
#     i = packet_detector(iq)[0]
#     packet = receiver(iq[i - 2:])
#
# l()


class bits(str):
    def __new__(cls, val: str):
        if val[0:2] in ('0x', '0X'):
            val = val[2:]
            num_of_bits = int(len(val) * np.log2(16))
            val = int_to_binstr(int(val, 16), num_of_bits)
            val = flip_byte_endian(val)  # IEE802.11 examples need this?!, tho it is confusing

        r = str.__new__(cls, val)
        return r

    # def __getitem__(self, item):
    #     res = super().__getitem__(item)
    #     return bits(res)

    # def __xor__(self, other):



a = bits('1010')
b = a[0]
pass