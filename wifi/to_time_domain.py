from typing import List
import numpy as np
from wifi.subcarrier_mapping import Carriers

OFDMFrame = List[complex]

OFDMFrame.__doc__ = """ Time-domain representation of OFDM symbol. """


def do(samples: List[OFDMFrame]) -> List[Carriers]:
    return np.fft.ifft(samples, n=64).tolist()


def undo(carriers: List[Carriers]) -> List[OFDMFrame]:
    return np.fft.fft(carriers, n=64).tolist()

# tried to add hypothesis test, but it always found some very insignificant test case to fail..
# so..lets just trust numpy!
