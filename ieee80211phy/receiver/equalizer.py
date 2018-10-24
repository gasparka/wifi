from ieee80211phy.transmitter.ofdm_modulation import demap_from_carriers, get_derotated_pilots
from ieee80211phy.transmitter.preamble import long_train_symbol
import numpy as np
import matplotlib.pyplot as plt


class Equalizer:
    def __init__(self, sample_advance):
        self.sample_advance = sample_advance
        self.equalizer_coefs = None

    def train(self, rx, start_of_long_training, debug=False):
        ideal_long = long_train_symbol()
        base_index = start_of_long_training + self.sample_advance + 32 + 64
        first_train = rx[base_index-64:base_index]
        second_train = rx[base_index:base_index + 64]
        avg_train = (first_train + second_train) / 2

        channel_estimate = np.fft.fft(avg_train) / np.fft.fft(ideal_long)
        self.equalizer_coefs = 1 / channel_estimate

        if debug:
            plt.figure(figsize=(9.75, 5))
            plt.plot(avg_train, label='avg')
            plt.plot(first_train, label='first')
            plt.plot(second_train, label='second')
            plt.plot(ideal_long, label='ideal')
            plt.tight_layout()
            plt.legend()
            plt.grid()

            plt.figure(figsize=(9.75, 5))
            plt.ylim([-60, 20])
            plt.plot(np.fft.fftshift(20 * np.log10(np.abs(self.equalizer_coefs))))
            plt.tight_layout()
            plt.grid()

    def apply(self, x):
        return x * self.equalizer_coefs


def demod_data(rx, start_of_long_training, n_symbols, equalizer, debug=False):
    data_rx = rx[start_of_long_training+160:start_of_long_training+160 + (n_symbols*80)]

    groups = np.reshape(data_rx, (-1, 80))
    symbols = []
    pilots_debug = []
    for symbol_number, group in enumerate(groups):
        freq = np.fft.fft(group[16:])
        equalized = equalizer.apply(freq)
        pilots = get_derotated_pilots(equalized, symbol_number+1)
        pilots_debug.append(pilots)

        # # fix angle
        parasitic_angle = np.angle(np.mean(pilots))
        equalized = equalized * np.exp(-1j*parasitic_angle)
        equalizer.equalizer_coefs *= np.exp(-1j*parasitic_angle)

        symbols.append(demap_from_carriers(equalized))

    if debug:
        pilots_debug = np.array(pilots_debug)
        plt.figure(figsize=(9.75, 5))
        plt.plot(np.angle(pilots_debug.T[0]), label='pilot -21')
        plt.plot(np.angle(pilots_debug.T[1]), label='pilot -7')
        plt.plot(np.angle(pilots_debug.T[2]), label='pilot 7')
        plt.plot(np.angle(pilots_debug.T[3]), label='pilot 21')
        # plt.plot(avg, label='average')
        plt.tight_layout()
        plt.legend()
        plt.grid()

    return symbols


def phase_drift(symbols):
    pass