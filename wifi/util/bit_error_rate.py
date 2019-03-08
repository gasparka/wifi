from multiprocessing import Pool
from tqdm import tqdm
from wifi import transceiver
import numpy as np
from wifi.util.util import awgn


def _ber(iq, snr, expected_bits):
    iq = awgn(iq, snr)
    try:
        actual_bits = transceiver.undo(iq)
    except:
        return 1.0

    diff = len([True for x, y in zip(expected_bits, actual_bits) if x != y])
    return diff / len(expected_bits)


def do(data, data_rate, snr_range, averaging=256):
    iq = transceiver.do(data, data_rate)
    with Pool(8) as p:
        ret = {}
        for snr in tqdm(snr_range):
            res = p.starmap(_ber, [(iq, snr, data)] * averaging)
            ret[snr] = np.mean(res)
    return ret

