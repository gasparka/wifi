from ieee80211phy.util import moving_average, mixer
import numpy as np
import matplotlib.pyplot as plt


def fix_frequency_offset_coarse(rx, start_of_long_training, debug=False):
    # coarse offset - quite unreliable on noisy signal
    autocorrelation = rx[:-16] * np.conjugate(rx[16:])
    avg = moving_average(autocorrelation, 32)
    angle = np.angle(avg)
    angle = moving_average(angle, 64)

    freq_error = angle * (20e6 / (2 * np.pi * 16))

    index = start_of_long_training-42
    freq_error_selected = freq_error[index]
    print(f'Coarse freq error is {freq_error_selected}')

    fixed_rx = mixer(rx, freq_error_selected, 20e6)

    if debug:
        plt.figure(figsize=(9.75, 5))
        plt.plot(freq_error, label='freq_err')
        plt.stem([index], [freq_error_selected])
        plt.ylim([-freq_error_selected*2, freq_error_selected*2])
        plt.xlim([start_of_long_training-160, start_of_long_training+40])
        plt.legend()
        plt.tight_layout()
        plt.grid()

    return fixed_rx, freq_error_selected


def fix_frequency_offset_fine(rx, start_of_long_training, debug=False):
    input = rx
    autocorrelation = input[:-64] * np.conjugate(input[64:])
    angle = moving_average(autocorrelation, 64)
    angle = np.angle(angle)

    freq_error = angle * (20e6 / (2 * np.pi * 64))

    index = start_of_long_training+84
    freq_error_selected = freq_error[index]
    print(f'Fine freq error is {freq_error_selected}')

    fixed_rx = mixer(input, freq_error_selected, 20e6)

    if debug:
        plt.figure(figsize=(9.75, 5))
        plt.plot(freq_error, label='freq_err')
        plt.stem([index], [freq_error_selected])
        # plt.ylim([-2000, 1000])
        # plt.xlim([start_of_long_training, start_of_long_training+160])
        plt.legend()
        plt.tight_layout()
        plt.grid()

    return fixed_rx, freq_error_selected