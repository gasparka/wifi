from ieee80211phy.receiver.packet_detector import packet_detector
from ieee80211phy.transmitter.ofdm_modulation import extract_symbols, get_derotated_pilots
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

        # iq, error_coarse = fix_frequency_offset_coarse(iq, start_of_long_training, debug=True)

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
        data_rx = iq[start_of_long_training + 160:start_of_long_training + 160 + (n_symbols * 80)]

        # parse the signal field
        signal_field = data_rx[:80]
        start = 16 + self.sample_advance
        symbols = np.fft.fft(signal_field[start:start + 64])
        equalized_symbols = symbols * self.equalizer_taps



        data = np.reshape(data_rx[80:], (-1, 80))
        output_symbols = []

        correction = [0 + 0j] * 8
        debug_error=[]
        for symbol_number, group in enumerate(data):
            # 1. strip GI and get symbols
            start = 16 + self.sample_advance
            symbols = np.fft.fft(group[start:start + 64])

            # 2. equalize the symbols
            equalized_symbols = symbols * self.equalizer_taps

            # 3. remove latent frequency offset by using pilot symbols
            # i.e. this is the average phase offset that
            pilots = get_derotated_pilots(equalized_symbols, symbol_number + 1)
            mean_phase_offset = np.angle(np.mean(pilots))
            equalized_symbols *= np.exp(-1j * mean_phase_offset)
            self.equalizer_taps *= np.exp(-1j * mean_phase_offset)

            # output
            output_symbols.append(extract_symbols(equalized_symbols))

        return output_symbols

    def plot_channel_estimate(self):
        """
        Rising trend in the phase response is due to the 'sample_advance' parameter.
        """

        fig, ax = plt.subplots(2, figsize=(9.75, 6), sharex='all')
        plt.title('Equalizers effect to each carrier')
        taps = np.fft.fftshift(self.equalizer_taps)
        mag = 20 * np.log10(taps)
        deg = np.degrees(np.unwrap(np.angle(taps)))
        ax[0].set_title('Magnitude response')
        ax[0].plot(np.arange(-32, 32), mag)
        ax[0].set_ylim([mag.max()-12, mag.max()+6])
        ax[0].set_ylabel('Magnitude dB')
        ax[0].grid(True)
        ax[1].set_title('Phase response (unwrapped)')
        ax[1].plot(np.arange(-32, 32), deg)
        ax[1].set_ylabel('Angle [deg]')
        ax[1].set_xlabel('Carrier index')
        ax[1].grid(True)
        plt.tight_layout()


def test_limemini_wire_loopback():
    """
    This was recorded with single LimeSDR-Mini, Tx port was connected to Rx with a cable.
    Gains were quite high. Looks like this setup has no frequency offset or timing drift.
    """
    iq = np.load('/home/gaspar/git/ieee80211phy/data/limemini_wire_loopback.npy')
    r = Receiver(sample_advance=-3)
    symbols = r.main(iq, n_symbols=109)

    evm = evm_db(symbols, default_reference_symbols()[1:])
    assert int(evm) == -30


def test_limemini_air():
    """
    Similar to previous, but over the air.
    """

    iq = np.load('/home/gaspar/git/ieee80211phy/data/limemini_air.npy')
    r = Receiver(sample_advance=-3)
    symbols = r.main(iq, n_symbols=229)

    from ieee80211phy.transmitter.subcarrier_modulation_mapping import mapper_decide
    reference_symbols = np.array([[mapper_decide(j, 4) for j in x] for x in symbols])
    evm = evm_db(symbols, reference_symbols)
    assert int(evm) == -25


def test_limemini_lime_air():
    """
    Lime-Mini as TX and Lime as RX. First test with different devices, resulted in frequency offset - that
    was best corrected using the pilot symbols!
    """

    iq = np.load('/home/gaspar/git/ieee80211phy/data/limemini_lime_air.npy')
    r = Receiver(sample_advance=-3)
    symbols = r.main(iq, n_symbols=229)

    from ieee80211phy.transmitter.subcarrier_modulation_mapping import mapper_decide
    reference_symbols = np.array([[mapper_decide(j, 4) for j in x] for x in symbols])
    evm = evm_db(symbols, reference_symbols)
    assert int(evm) == -26
