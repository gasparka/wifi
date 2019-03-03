# """
# Uses the 'short training sequence' i.e. the first 160 samples of the WiFi packet to detect valid packets.
# """
#
# import numpy as np
# from hypothesis import given, settings
# from hypothesis._strategies import integers
# from wifi.util import moving_average, mixer, awgn
#
# trace = {}
#
#
# def do(iq):
#     """
#     Generator
#     Returns index to start of first long training symbol
#     """
#     acorr = (iq[:-16] * np.conjugate(iq[16:])).real
#     acorr = moving_average(acorr, 8)
#     trace['autocorrelation'] = acorr / acorr.max()
#
#     power = (iq * np.conjugate(iq)).real
#     power = moving_average(power, 8)
#     trace['power'] = power / power.max()
#
#     ratio = [1.0 if x else -1.0 for x in acorr >= 0.5 * power[:len(acorr)]]
#     ratio = moving_average(ratio, 128)  # valid packet will have high ratio
#     trace['detectionratio'] = ratio
#
#     try:
#         high = np.where(ratio >= 0.8)[0][0]  # first point going high
#         low = np.where(ratio[high:] < 0.6)[0][0]  # first point going low, after being high
#         return high + low - 172  # - 177 shifts the detection index to package start
#     except IndexError:
#         return -1
#
#
# @settings(deadline=None, max_examples=1024)
# @given(integers(min_value=6, max_value=30))
# def test_detect(snr):
#     iq = get()
#
#     # add 128 random samples in front of the preamble - then test that detector finds it
#     pad_len = 128
#     noise = 0.01 * (np.random.randn(pad_len) + np.random.randn(pad_len) * 1j)
#     iq = np.concatenate([noise, iq])
#
#     def channel(tx, snr, freq_offset):
#         freq_offset = mixer(tx, freq_offset, 20e6)
#         rx = awgn(freq_offset, snr)
#         return rx
#
#     # CHANNEL
#     ch = channel(iq, snr, 0)
#
#     rx = detect(ch)
#     assert rx == 128
