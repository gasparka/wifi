# from ieee80211phy.receiver import packet_detector, receiver
from wifi.bits import bits
from wifi.util import bit_error_rate
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

# a = bits('123')
# assert isinstance(a, str)
# pass
