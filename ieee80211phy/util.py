import numpy as np
import matplotlib.pyplot as plt


def power(x):
    return (x * np.conjugate(x)).real


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


def plot_rx(rx_symbols, reference_symbols):
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
    ax[1].set(title=f'EVM vs carriers')
    ax[1].plot(evm_carrier)
    ax[1].grid(True)
    ax[1].set_xlabel('Carrier')
    ax[1].set_ylabel('EVM')

    # evm vs time
    evm_time = evm_vs_time(rx_symbols, reference_symbols)
    ax[2].set(title=f'EVM vs time')
    ax[2].plot(evm_time)
    ax[2].grid(True)
    ax[2].set_xlabel('OFDM packet')
    ax[2].set_ylabel('EVM')

    plt.tight_layout(rect=[0, 0, 1, 0.95])
    plt.show()
