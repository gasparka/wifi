from ieee80211phy.receiver.packet_detector import packet_detector
from ieee80211phy.transmitter.ofdm_modulation import demap_from_carriers
from ieee80211phy.transmitter.preamble import long_train_symbol
import numpy as np
import matplotlib.pyplot as plt

from ieee80211phy.util import evm_db, default_reference_symbols


class Receiver:
    def __init__(self, sample_advance=-3):
        self.sample_advance = sample_advance
        self.equalizer_taps = None
        pass

    def main(self, iq, n_symbols):
        start_of_long_training, _ = packet_detector(iq)

        # train the equalizer by using the long training symbols
        ideal_long_freq = np.fft.fft(long_train_symbol())
        base_index = start_of_long_training + self.sample_advance + 32
        first_train = iq[base_index:base_index + 64]
        second_train = iq[base_index + 64:base_index + 128]
        avg_train = (first_train + second_train) / 2
        avg_train_freq = np.fft.fft(avg_train)
        # avg_train_freq[-20] *= np.exp(-1j * np.deg2rad(30))
        # avg_train_freq[-20] *= 0.25
        channel_estimate = avg_train_freq / ideal_long_freq
        self.equalizer_taps = 1 / channel_estimate

        # parse the payload
        # TODO: should skip signal field here?
        data_rx = iq[start_of_long_training + 160:start_of_long_training + 160 + (n_symbols * 80)]
        groups = np.reshape(data_rx, (n_symbols, 80))
        output_symbols = []
        for symbol_number, group in enumerate(groups):
            # 1. strip GI and get symbols
            start = 16 + self.sample_advance
            symbols = np.fft.fft(group[start:start + 64])

            # 2. equalize the symbols
            equalized_symbols = symbols * self.equalizer_taps

            # output
            output_symbols.append(demap_from_carriers(equalized_symbols))

        return output_symbols

    def plot_channel_estimate(self):
        """

        """

        plt.figure(figsize=(9.75, 5))
        plt.title('Equalizers effect to each carrier')
        taps = np.fft.fftshift(self.equalizer_taps)
        mag = 20 * np.log10(taps)
        deg = np.degrees(np.angle(taps))
        plt.plot(np.arange(-32, 32), mag, label='magnitude [dB]')
        plt.plot(np.arange(-32, 32), deg, label='angle [deg]')
        plt.ylim([-60, max(mag.max(), deg.max())])
        plt.tight_layout()
        plt.legend()
        plt.grid()


def test_limemini_wire_loopback():
    iq = np.load('/home/gaspar/git/ieee80211phy/data/limemini_wire_loopback.npy')
    r = Receiver(sample_advance=-3)
    symbols = r.main(iq, n_symbols=109)

    evm = evm_db(symbols, default_reference_symbols())
    assert int(evm) == -30
