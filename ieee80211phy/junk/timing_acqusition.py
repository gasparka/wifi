import scipy
import numpy as np
import matplotlib.pyplot as plt


def timing_acquisition(rx, start_of_long_training, debug=False):
    long_symbol = np.array(long_train_symbol())
    corr = scipy.signal.correlate(rx, long_symbol)
    corr_sign = scipy.signal.correlate(rx, np.sign(long_symbol))
    corr_sign /= corr_sign.max() / corr.max()

    # skip over the first 'halfpeak' - whcih is from guard interval!
    offset = start_of_long_training + 54
    first_peak = np.argmax(corr[offset: offset + 64]) + offset
    offset += 64
    second_peak = np.argmax(corr[offset: offset + 64]) + offset
    print(first_peak)

    print(np.angle(corr[first_peak] * np.conjugate(corr[second_peak])) * (20e6/(2*np.pi*64)))
    print(np.angle(corr_sign[first_peak] * np.conjugate(corr_sign[second_peak])) * (20e6/(2*np.pi*64)))

    if debug:

        plt.figure(figsize=(9.75, 5))
        plt.plot(corr, label='corr')
        plt.plot(corr_sign, label='sign')
        plt.scatter([first_peak], [corr[first_peak]], label='first_peak_pos')
        plt.scatter([second_peak], [corr[second_peak]], label='second_peak_pos')
        plt.xlim([first_peak - 64, first_peak + 128])
        # plt.xlim([0, 500])
        plt.legend()
        plt.tight_layout()
        plt.grid()

        plt.figure(figsize=(9.75, 5))
        plt.plot(long_symbol / long_symbol.max(), label='ideal_long')
        long = rx[first_peak - 64+1:first_peak+1]
        plt.plot(long / long.max(), label='detected_long')
        plt.legend()
        plt.tight_layout()
        plt.grid()
