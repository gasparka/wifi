from multiprocessing import Pool
from tqdm import tqdm
from util.util import awgn
from wifi import transceiver
import numpy as np


def _per(iq, snr, expected_bits):
    iq = awgn(iq, snr)
    try:
        actual_bits = transceiver.undo(iq)
    except:
        return False

    return expected_bits == actual_bits


def do(data, data_rate, snr_range, averaging=256):
    iq = transceiver.do(data, data_rate)
    with Pool(8) as p:
        ret = {}
        for snr in tqdm(snr_range):
            res = p.starmap(_per, [(iq, snr, data)] * averaging)
            ret[snr] = np.mean(np.array(res).astype(float)) * 100
    return ret
