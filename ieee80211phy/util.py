import logging

import numpy as np
import matplotlib.pyplot as plt
import scipy
from scipy import signal

logging.basicConfig(level=logging.DEBUG)
log = logging.getLogger('util')

def default_reference_symbols():
    """ QAM16 symbols sent by the examples in the IEE802.11 document """
    return np.load('/home/gaspar/git/ieee80211phy/data/default_reference_symbols.npy')


def power(x):
    return (x * np.conjugate(x)).real


def SQNR(pure, noisy):
    sig_pow = np.mean(np.abs(pure))
    error = np.array(pure) - np.array(noisy)
    err_pow = np.mean(np.abs(error))

    snr_db = 20 * np.log10(sig_pow / err_pow)
    return snr_db


def awgn(iq, snr):
    mean_square = np.mean((iq * np.conjugate(iq)).real)
    noise_power = mean_square / (10 ** (snr / 10))
    std_noise = np.sqrt(noise_power)
    noise = std_noise * (0.70711 * np.random.randn(len(iq)) + 0.70711 * np.random.randn(len(iq)) * 1j)

    return noise + iq


def timing_offseset(tx, delay):
    in_index = np.arange(0, len(tx), 1)
    out_index = np.arange(0, len(tx), 1 + delay)
    print(f'Max err: {delay*len(tx)}')

    wat = scipy.interpolate.interp1d(in_index, tx, kind='slinear', fill_value='extrapolate')
    return wat(out_index)


def moving_average(inputs, window_len):
    taps = [1 / window_len] * window_len
    return signal.lfilter(taps, [1.0], inputs)

def moving_average_valid(inputs, window_len):
    """
    Skips the initial transient.
    """
    inputs = np.concatenate([[inputs[0]] * window_len, inputs])
    return moving_average(inputs, window_len)[window_len:]


def mixer(signal, lo_freq, fs):
    phase_inc = 2 * np.pi * lo_freq / fs
    phase_list = np.array(range(len(signal))) * phase_inc
    lo = np.exp(phase_list * 1j)

    mixed = signal * lo
    return mixed


def evm_db(rx, reference):
    error = rx - reference
    error_power = np.mean([power(e) for e in error])

    # reference power is known to be 1.0 for all constellations for IEE802.11
    evm_db = 10 * np.log10(error_power)
    # evm_rms = np.sqrt(error_power) * 100
    return evm_db


def evm_vs_carrier(rx, ref):
    return [evm_db(crx, cref) for crx, cref in zip(rx.T, ref.T)]


def evm_vs_time(rx, ref):
    return [evm_db(crx, cref) for crx, cref in zip(rx, ref)]


def plot_rx(rx_symbols, reference_symbols=None):
    rx_symbols = np.array(rx_symbols)
    if reference_symbols is None:
        log.warning('Using decicion for reference symbols! EVM may be misleading!')
        print(rx_symbols)
        from ieee80211phy.transmitter.subcarrier_modulation_mapping import mapper_decide
        reference_symbols = np.array([[mapper_decide(j, 4) for j in x] for x in rx_symbols]) #TODO: remove this shit code
        print(reference_symbols)

    figsize = (9.75, 10)
    fig, ax = plt.subplots(3, figsize=figsize, gridspec_kw={'height_ratios': [4, 2, 2]})

    # constellation
    ax[0].set(title=f'Constellation EVM={evm_db(rx_symbols, reference_symbols):.2f} dB')
    ax[0].scatter(rx_symbols.real, rx_symbols.imag)
    ax[0].grid(True)
    tick_base = 1 / np.sqrt(10)
    ax[0].set_xticks([-4 * tick_base, -2 * tick_base, 0, tick_base * 2, tick_base * 4])
    ax[0].set_yticks([-4 * tick_base, -2 * tick_base, 0, tick_base * 2, tick_base * 4])
    ax[0].set_xlabel('Real')
    ax[0].set_ylabel('Imag')

    # evm vs carrier
    evm_carrier = evm_vs_carrier(rx_symbols, reference_symbols)
    ids = [-26, -25, -24, -23, -22,
           -20, -19, -18, -17, -16, -15, -14, -13, -12, -11, -10, -9, -8,
           -6, -5, -4, -3, -2, -1,
           1, 2, 3, 4, 5, 6,
           8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20,
           22, 23, 24, 25, 26]

    ax[1].set(title=f'EVM vs carriers')
    ax[1].scatter(ids, evm_carrier)
    ax[1].set_xticks(ids)
    ax[1].set_xticklabels(ids, rotation=45)
    ax[1].grid(True)
    ax[1].set_xlabel('Carrier')
    ax[1].set_ylabel('EVM')

    # evm vs time
    evm_time = evm_vs_time(rx_symbols, reference_symbols)
    ax[2].set(title=f'EVM vs time')
    ax[2].plot(evm_time)
    ax[2].plot(moving_average_valid(evm_time, 32), alpha=0.5)
    ax[2].grid(True)
    ax[2].set_xlabel('OFDM packet')
    ax[2].set_ylabel('EVM')

    plt.tight_layout(rect=[0, 0, 1, 0.95])
    plt.show()
