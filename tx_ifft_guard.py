import numpy as np

def to_time_domain(symbols):
    """ Takes 48 input symbols in Freq domain and outputs 80 outputs in time domain.
    Includes Guard interval, pilot tones and padding symbols"""

    ifft_in = np.zeros(64)
    indexes = []
