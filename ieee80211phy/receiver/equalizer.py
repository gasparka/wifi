from ieee80211phy.transmitter.ofdm_modulation import extract_symbols, get_derotated_pilots
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
        first_train = rx[base_index - 64:base_index]
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


def demod_data(rx, start_of_long_training, n_symbols, debug=False, fix_mean_pilots_phase=True):
    equalizer = Equalizer(sample_advance=0)
    equalizer.train(rx, start_of_long_training, debug=False)

    data_rx = rx[start_of_long_training + 160:start_of_long_training + 160 + (n_symbols * 80)]

    groups = np.reshape(data_rx, (-1, 80))
    output_symbols = []
    pilots_debug = []
    slope_debug = []
    mean_phase_offset_debug = []
    for symbol_number, group in enumerate(groups):
        # 1. strip GI and get symbols
        symbols = np.fft.fft(group[16:])

        # 2. equalize the symbols
        equalized_symbols = equalizer.apply(symbols)

        # 3. remove latent frequency offset by using pilot symbols
        # i.e. this is the average phase offset that
        pilots = get_derotated_pilots(equalized_symbols, symbol_number + 1)
        pilots_debug.append(pilots)
        mean_phase_offset = np.angle(np.mean(pilots))
        mean_phase_offset_debug.append(mean_phase_offset)
        if fix_mean_pilots_phase:
            equalized_symbols *= np.exp(-1j * mean_phase_offset)
            equalizer.equalizer_coefs *= np.exp(-1j * mean_phase_offset)

        # decision feedback equalizer

        # conj_piltos = pilots * np.conjugate(mean_phase_offset)
        # pilots = np.angle(conj_piltos)
        # # pilots = np.angle(pilots)
        # pilots /= np.array([-21, -7, 7, 21])
        # slope = np.mean(pilots)
        #
        # coef = [0.0] * 64
        # for x in np.arange(-32, 32):
        #     coef[x] = np.exp(-1j * ((slope * x) + avg_pilots_angle))
        # # pil2_debug.append(coef)
        #
        # equalizer.equalizer_coefs *= coef
        # equalized_symbols *= coef

        # # fix angle drift
        # mean_pilots = np.mean(pilots)
        # parasitic_angle = np.angle(mean_pilots)
        # equalized = equalized * np.exp(-1j*parasitic_angle)
        # equalizer.equalizer_coefs *= np.exp(-1j*parasitic_angle)

        # # fix timing drift
        # # slopes = pilots * np.conjugate(mean_pilots)
        # # slopes = np.angle(slopes)
        # slopes = np.angle(pilots)
        # slope = (-slopes[0]/21 - slopes[1]/7 + slopes[2]/7 + slopes[3]/21) / 4
        # coefs = np.exp(-1j * (np.arange(-32, 32) * slope))
        # for i in np.arange(-32, 32):
        #     equalized[i] = equalized[i] * coefs[i]
        #     equalizer.equalizer_coefs[i] = equalizer.equalizer_coefs[i] * coefs[i]
        # # equalized = equalized * coefs
        # # equalizer.equalizer_coefs *= coefs
        # slope_debug.append(slope)

        output_symbols.append(extract_symbols(equalized_symbols))

    if debug:
        # pilots_debug = np.array(pilots_debug)

        # plt.figure(figsize=(9.75, 5))
        # for x in pilots_debug:
        #     l = x[0] * np.conjugate(np.mean(x))
        #     ll = x[0] - np.mean(x)
        #     plt.scatter(l.imag, l.real)
        #     plt.scatter(ll.imag, ll.real)
        #
        # plt.tight_layout()
        # plt.legend()
        # plt.grid()

        # slope_debug = np.array(slope_debug)

        plt.figure(figsize=(9.75, 5))
        plt.title('Pilots mean phase offset')
        plt.plot(mean_phase_offset_debug)
        plt.tight_layout()
        plt.legend()
        plt.grid()

        # plt.figure(figsize=(9.75, 5))
        # plt.plot(np.angle(pilots_debug.T[0]), label='pilot -21')
        # plt.plot(np.angle(pilots_debug.T[1]), label='pilot -7')
        # plt.plot(np.angle(pilots_debug.T[2]), label='pilot 7')
        # plt.plot(np.angle(pilots_debug.T[3]), label='pilot 21')
        # # avg = (-np.angle(pilots_debug.T[0])/21 - np.angle(pilots_debug.T[1])/7 + np.angle(pilots_debug.T[2])/7 + np.angle(pilots_debug.T[3])/21) / 4
        # # plt.plot(avg, label='average')
        # plt.tight_layout()
        # plt.legend()
        # plt.grid()

    return output_symbols
