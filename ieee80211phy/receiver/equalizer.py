from ieee80211phy.transmitter.preamble import long_train_symbol
import numpy as np
import matplotlib.pyplot as plt


class Equalizer:
    def __init__(self, sample_advance):
        self.sample_advance = sample_advance
        self.equalizer_coefs = None

    def train(self, rx, start_of_training_symbol, debug=False):
        ideal_long = long_train_symbol()
        first_train = rx[start_of_training_symbol + self.sample_advance-64:start_of_training_symbol + self.sample_advance]
        second_train = rx[start_of_training_symbol + self.sample_advance : start_of_training_symbol + self.sample_advance + 64]
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