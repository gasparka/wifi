import logging
import numpy as np

logging.basicConfig(level=logging.DEBUG)
log = logging.getLogger('util')


def is_divisible(x, by):
    return len(x) != 0 and len(x) % by == 0



class Bits(str):
    def __new__(cls, val: str):
        if val[0:2] in ('0x', '0X'):
            val = val[2:]
            num_of_bits = int(len(val) * np.log2(16))
            val = int_to_binstr(int(val, 16), num_of_bits)
            val = flip_byte_endian(val)  # IEE802.11 examples need this?!, tho it is confusing

        return str.__new__(cls, val)


def default_iee80211_package():
    # Table I-1—The message for the BCC example
    input = Bits('0x0402002E006008CD37A60020D6013CF1006008AD3BAF00004A6F792C2062726967687420737061726B206F662064'
                 '6976696E6974792C0A4461756768746572206F6620456C797369756D2C0A466972652D696E73697265642077652074'
                 '726561673321B60402002E006008CD37A60020D6013CF1006008AD3BAF00004A6F792C2062726967687420737061726B206F662064'
                 '6976696E6974792C0A4461756768746572206F6620456C797369756D2C0A466972652D696E73697265642077652074'
                 '726561673321B60402002E006008CD37A60020D6013CF1006008AD3BAF00004A6F792C2062726967687420737061726B206F662064'
                 '6976696E6974792C0A4461756768746572206F6620456C797369756D2C0A466972652D696E73697265642077652074'
                 '726561673321B60402002E006008CD37A60020D6013CF1006008AD3BAF00004A6F792C2062726967687420737061726B206F662064'
                 '6976696E6974792C0A4461756768746572206F6620456C797369756D2C0A466972652D696E73697265642077652074'
                 '726561673321B60402002E006008CD37A60020D6013CF1006008AD3BAF00004A6F792C2062726967687420737061726B206F662064'
                 '6976696E6974792C0A4461756768746572206F6620456C797369756D2C0A466972652D696E73697265642077652074'
                 '726561673321B60402002E006008CD37A60020D6013CF1006008AD3BAF00004A6F792C2062726967687420737061726B206F662064'
                 '6976696E6974792C0A4461756768746572206F6620456C797369756D2C0A466972652D696E73697265642077652074'
                 '726561673321B60402002E006008CD37A60020D6013CF1006008AD3BAF00004A6F792C2062726967687420737061726B206F662064'
                 '6976696E6974792C0A4461756768746572206F6620456C797369756D2C0A466972652D696E73697265642077652074'
                 '726561673321B60402002E006008CD37A60020D6013CF1006008AD3BAF00004A6F792C2062726967687420737061726B206F662064'
                 '6976696E6974792C0A4461756768746572206F6620456C797369756D2C0A466972652D696E73697265642077652074'
                 '726561673321B60402002E006008CD37A60020D6013CF1006008AD3BAF00004A6F792C2062726967687420737061726B206F662064'
                 '6976696E6974792C0A4461756768746572206F6620456C797369756D2C0A466972652D696E73697265642077652074'
                 '726561673321B60402002E006008CD37A60020D6013CF1006008AD3BAF00004A6F792C2062726967687420737061726B206F662064'
                 '6976696E6974792C0A4461756768746572206F6620456C797369756D2C0A466972652D696E73697265642077652074'
                 '726561673321B60402002E006008CD37A60020D6013CF1006008AD3BAF00004A6F792C2062726967687420737061726B206F662064'
                 '6976696E6974792C0A4461756768746572206F6620456C797369756D2C0A466972652D696E73697265642077652074'
                 '726561673321B6')

    from wifi.transmitter import transmit
    return input, transmit(Bits(input), data_rate=24)


def bit_error_rate(snr):
    from wifi.receiver import receiver
    bits, iq = default_iee80211_package()

    def one(iq):
        iq = awgn(iq, snr)
        try:
            res = receiver(iq[160 + 30:])
        except:
            return 1.0
        diff = len([True for x, y in zip(bits, res.bits) if x != y])
        return diff / len(bits)

    diffs = [one(iq) for _ in range(32)]

    return np.mean(diffs)


def hex_to_bitstr(hstr):
    """ http://stackoverflow.com/questions/1425493/convert-hex-to-binary """
    assert isinstance(hstr, str)
    if hstr[0:2] in ('0x', '0X'):
        hstr = hstr[2:]
    my_hexdata = hstr
    num_of_bits = int(len(my_hexdata) * np.log2(16))
    return int_to_binstr(int(my_hexdata, 16), num_of_bits)


def flip_byte_endian(bitstr):
    from textwrap import wrap
    bytes = wrap(bitstr, 8)
    flipped = [x[::-1] for x in bytes]
    return ''.join(flipped)


def reverse(x: str):
    return x[::-1]


def xor_reduce_poly(data, poly):
    """ XOR reduces bits that are selected by the 'poly' """
    return int(bin(data & poly).count('1') & 1)


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


def timing_offset(tx, delay):
    in_index = np.arange(0, len(tx), 1)
    out_index = np.arange(0, len(tx), 1 + delay)
    print(f'Max err: {delay*len(tx)}')

    wat = scipy.interpolate.interp1d(in_index, tx, kind='slinear', fill_value='extrapolate')
    return wat(out_index)


def moving_average(inputs, window_len):
    from scipy import signal
    taps = [1 / window_len] * window_len
    return signal.lfilter(taps, [1.0], inputs)


def moving_average_valid(inputs, window_len):
    """
    Skips the initial transient.
    """
    inputs = np.concatenate([[inputs[0]] * (window_len // 2), inputs, [inputs[-1]] * (window_len // 2)])
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


def evm_db2(rx, bits_per_symbol):
    from wifi.modulator import symbols_error
    rx = np.array(rx).flatten()
    error = symbols_error(rx, bits_per_symbol)
    error_power = np.mean([power(e) for e in error])

    # reference power is known to be 1.0 for all constellations for IEE802.11
    evm_db = 10 * np.log10(error_power)
    # evm_rms = np.sqrt(error_power) * 100
    return evm_db


def evm_vs_carrier(rx, ref):
    return [evm_db2(crx, 4) for crx in rx.T]


def evm_vs_time(rx, ref):
    return [evm_db(crx, cref) for crx, cref in zip(rx, ref)]


def plot_rx(rx_symbols, bits_per_symbol):
    rx_symbols = np.array(rx_symbols)
    figsize = (9.75, 15)
    fig, ax = plt.subplots(3, figsize=figsize, gridspec_kw={'height_ratios': [4, 2, 2]})

    # constellation
    rx_constellation = np.array(rx_symbols).flatten()
    evm = evm_db2(rx_constellation, bits_per_symbol)
    ax[0].set(title=f'Constellation EVM={evm:.2f} dB')
    ax[0].scatter(rx_constellation.real, rx_constellation.imag)
    ax[0].grid(True)

    if bits_per_symbol == 4:
        tick_base = 1 / np.sqrt(10)
        ax[0].set_xticks([-4 * tick_base, -2 * tick_base, 0, tick_base * 2, tick_base * 4])
        ax[0].set_yticks([-4 * tick_base, -2 * tick_base, 0, tick_base * 2, tick_base * 4])
    elif bits_per_symbol == 6:
        tick_base = 1 / np.sqrt(42)
        ax[0].set_xticks(
            [-8 * tick_base, -6 * tick_base, -4 * tick_base, -2 * tick_base, 0, tick_base * 2, tick_base * 4,
             6 * tick_base, 8 * tick_base])
        ax[0].set_yticks(
            [-8 * tick_base, -6 * tick_base, -4 * tick_base, -2 * tick_base, 0, tick_base * 2, tick_base * 4,
             6 * tick_base, 8 * tick_base])
    elif bits_per_symbol == 1:
        tick_base = 1
        ax[0].scatter([-1, 1], [0, 0], marker='x')
        ax[0].set_xticks([-2 * tick_base, 0, tick_base * 2])
        ax[0].set_yticks([-2 * tick_base, tick_base * 2])
    elif bits_per_symbol == 2:
        tick_base = 1 / np.sqrt(2)
        ax[0].scatter([tick_base, tick_base, -tick_base, -tick_base], [tick_base, -tick_base, tick_base, -tick_base],
                      marker='x')
        ax[0].set_xticks([-2 * tick_base, 0, tick_base * 2])
        ax[0].set_yticks([-2 * tick_base, 0, tick_base * 2])

    ax[0].set_xlabel('Real')
    ax[0].set_ylabel('Imag')

    # # evm vs carrier
    evm_carrier = [evm_db2(crx, bits_per_symbol) for crx in rx_symbols.T]
    ids = [-26, -25, -24, -23, -22,
           -20, -19, -18, -17, -16, -15, -14, -13, -12, -11, -10, -9, -8,
           -6, -5, -4, -3, -2, -1,
           1, 2, 3, 4, 5, 6,
           8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20,
           22, 23, 24, 25, 26]

    ax[1].set(title=f'EVM vs carriers')
    ax[1].scatter(ids, evm_carrier)
    ax[1].plot(ids, moving_average_valid(evm_carrier, 4), alpha=0.25)
    ax[1].set_xticks(ids)
    ax[1].set_xticklabels(ids, rotation=45)
    ax[1].grid(True)
    ax[1].set_xlabel('Carrier')
    ax[1].set_ylabel('EVM')

    # evm vs time
    evm_time = [evm_db2(crx, bits_per_symbol) for crx in rx_symbols]
    ax[2].set(title=f'EVM vs time')
    ax[2].plot(evm_time)
    ax[2].plot(moving_average_valid(evm_time, 32), alpha=0.25)
    ax[2].grid(True)
    ax[2].set_xlabel('OFDM packet')
    ax[2].set_ylabel('EVM')

    plt.tight_layout(rect=[0, 0, 1, 0.95])
    plt.show()


def plot_channel_estimate(equalizer):
    """
    Rising trend in the phase response is due to the 'sample_advance' parameter.
    """

    fig, ax = plt.subplots(2, figsize=(9.75, 6), sharex='all')
    taps = np.fft.fftshift(equalizer)
    mag = 20 * np.log10(np.abs(taps))
    deg = np.degrees(np.angle(taps))
    ax[0].set_title('Equalizer magnitude response')
    ax[0].plot(np.arange(-32, 32), mag)
    ax[0].set_ylim([np.nanmin(mag) - 12, np.nanmax(mag) + 6])
    ax[0].set_ylabel('Magnitude dB')
    ax[0].grid(True)
    ax[1].set_title('Equalizer phase response')
    ax[1].plot(np.arange(-32, 32), deg)
    ax[1].set_ylabel('Angle [deg]')
    ax[1].set_xlabel('Carrier index')
    ax[1].grid(True)
    plt.tight_layout()


def plot_packet_time_domain(iq, i):
    plt.figure(figsize=(9.75, 5))
    plt.title('Packet in time domain')
    plt.plot(np.arange(0, 160), iq[i - 160 - 32: i - 32], label='short training')
    plt.plot(np.arange(160, 320), iq[i - 32: i + 128], label='long training')
    plt.plot(np.arange(320, 400), iq[i + 128: i + 128 + 80], label='signal field')
    plt.plot(np.arange(400, 480), iq[i + 128 + 80: i + 128 + 80 + 80], label='first data')
    plt.scatter([i], [iq[i]], label='timing recovery index')

    plt.grid()
    plt.legend()
    plt.tight_layout()


def plot_signal_field_constellation(symbols):
    plt.figure(figsize=(9.75, 5))

    evm = evm_db2(symbols, bits_per_symbol=1)
    plt.title(f'Signal field constallation. EVM={evm:.2f} dB')
    plt.scatter(symbols.real, symbols.imag)
    plt.scatter([-1, 1], [0, 0], marker='x')
    plt.ylim([-1, 1])
    plt.xlim([-2, 2])

    a = plt.gca().get_xgridlines()
    b = a[4]
    b.set_color('red')
    b.set_linewidth(1)

    plt.tight_layout()
    plt.grid()
