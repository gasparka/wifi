import numpy as np

from ieee80211phy.transmitter.subcarrier_modulation_mapping import mapper


def insert_pilots_and_pad(symbols):
    """ Inserts 4 pilots at position -21, -7, 7, 21 and pads start and end with zeroes
    Also adds a zero for the zero carrier
    """
    # TODO: simplified to always use 1+0j pilots!
    pad_head = [0 + 0j] * 6
    pad_tail = [0 + 0j] * 5
    symbols = pad_head + list(symbols) + pad_tail
    symbols.insert(11, 1 + 0j)  # pilot at carrier -21
    symbols.insert(25, 1 + 0j)  # pilot at carrier -7
    symbols.insert(32, 0 + 0j)  # zero at carrier 0
    symbols.insert(39, 1 + 0j)  # pilot at carrier 7
    symbols.insert(53, -1 + 0j)  # pilot at carrier 21

    return np.array(symbols)


def ifft_guard(symbols):
    """ j) Divide the complex number string into groups of 48 complex numbers. Each such group is
        associated with one OFDM symbol. In each group, the complex numbers are numbered 0 to 47 and
        mapped hereafter into OFDM subcarriers numbered -26 to -22, -20 to -8, -6 to -1, 1 to 6, 8 to 20,
        and 22 to 26. The subcarriers -21, -7, 7, and 21 are skipped and, subsequently, used for inserting
        pilot subcarriers. The 0 subcarrier, associated with center frequency, is omitted and filled with the
        value 0. Refer to 17.3.5.10 for details.

        k) Four subcarriers are inserted as pilots into positions -21, -7, 7, and 21. The total number of the
        subcarriers is 52 (48 + 4). Refer to 17.3.5.9 for details.

        l) For each group of subcarriers -26 to 26, convert the subcarriers to time domain using inverse
        Fourier transform. Prepend to the Fourier-transformed waveform a circular extension of itself thus
        forming a GI, and truncate the resulting periodic waveform to a single OFDM symbol length by
        applying time domain windowing. Refer to 17.3.5.10 for details.

        Clarification: IFFT of 64 points are used. 802.11 uses 48 subcarriers + 4 pilot tones, thus leaving
        12 empty tones. This is solved by setting -32..-27, 0 and 27..31 subcarriers to 0+0j.

    """
    # symbols = insert_pilots_and_pad(symbols)
    # symbols = np.fft.fftshift(symbols)
    ifft = np.fft.ifft(symbols)
    result = np.concatenate([ifft[48:], ifft])
    return result


def map_to_carriers(symbol):
    carrier = np.zeros(64, dtype=np.complex64)
    carrier[-32] = 0
    carrier[-31] = 0
    carrier[-30] = 0
    carrier[-29] = 0
    carrier[-28] = 0
    carrier[-27] = 0
    carrier[-26] = symbol[0]
    carrier[-25] = symbol[1]
    carrier[-24] = symbol[2]
    carrier[-23] = symbol[3]
    carrier[-22] = symbol[4]
    carrier[-21] = 0
    carrier[-20] = symbol[5]
    carrier[-19] = symbol[6]
    carrier[-18] = symbol[7]
    carrier[-17] = symbol[8]
    carrier[-16] = symbol[9]
    carrier[-15] = symbol[10]
    carrier[-14] = symbol[11]
    carrier[-13] = symbol[12]
    carrier[-12] = symbol[13]
    carrier[-11] = symbol[14]
    carrier[-10] = symbol[15]
    carrier[-9] = symbol[16]
    carrier[-8] = symbol[17]
    carrier[-7] = 0
    carrier[-6] = symbol[18]
    carrier[-5] = symbol[19]
    carrier[-4] = symbol[20]
    carrier[-3] = symbol[21]
    carrier[-2] = symbol[22]
    carrier[-1] = symbol[23]
    carrier[0] = 0
    carrier[1] = symbol[24]
    carrier[2] = symbol[25]
    carrier[3] = symbol[26]
    carrier[4] = symbol[27]
    carrier[5] = symbol[28]
    carrier[6] = symbol[29]
    carrier[7] = 0
    carrier[8] = symbol[30]
    carrier[9] = symbol[31]
    carrier[10] = symbol[32]
    carrier[11] = symbol[33]
    carrier[12] = symbol[34]
    carrier[13] = symbol[35]
    carrier[14] = symbol[36]
    carrier[15] = symbol[37]
    carrier[16] = symbol[38]
    carrier[17] = symbol[39]
    carrier[18] = symbol[40]
    carrier[19] = symbol[41]
    carrier[20] = symbol[42]
    carrier[21] = 0
    carrier[22] = symbol[43]
    carrier[23] = symbol[44]
    carrier[24] = symbol[45]
    carrier[25] = symbol[46]
    carrier[26] = symbol[47]
    carrier[27] = 0
    carrier[28] = 0
    carrier[29] = 0
    carrier[30] = 0
    carrier[31] = 0

    return carrier


def insert_pilots(ofdm_symbol, i):
    """
    In each OFDM symbol, four of the subcarriers are dedicated to pilot signals in order to make the coherent
    detection robust against frequency offsets and phase noise. These pilot signals shall be put in subcarriers
    -21, -7, 7, and 21. The pilots shall be BPSK modulated by a pseudo-binary sequence to prevent the
    generation of spectral lines. The contribution of the pilot subcarriers to each OFDM symbol is described in
    17.3.5.10.
    
    :param ofdm_symbol:
    :param i: symbol position, note that SIGNAL field is pos 0
    :return:
    """

    polarity = [1, 1, 1, 1, -1, -1, -1, 1, -1, -1, -1, -1, 1, 1, -1, 1, -1, -1, 1, 1, -1, 1, 1, -1, 1, 1, 1, 1, 1, 1,
                -1, 1, 1, 1, -1, 1, 1, -1, -1, 1, 1, 1, -1, 1, -1, -1, -1, 1, -1, 1, -1, -1, 1, -1, -1, 1, 1, 1, 1, 1, -1, -1,
                1, 1, -1, -1, 1, -1, 1, -1, 1, 1, -1, -1, -1, 1, 1, -1, -1, -1, -1, 1, -1, -1, 1, -1, 1, 1, 1, 1, -1, 1, -1,
                1, -1, 1, -1, -1, -1, -1, -1, 1, -1, 1, 1, -1, 1, -1, 1, 1, 1, -1, -1, 1, -1, -1, -1, 1, 1, 1, -1, -1, -1, -1, -1,
                -1, -1]
    pilots = np.array([1, 1, 1, -1]) * polarity[i % 128]
    ofdm_symbol[-21] = pilots[0]
    ofdm_symbol[-7] = pilots[1]
    ofdm_symbol[7] = pilots[2]
    ofdm_symbol[21] = pilots[3]
    return ofdm_symbol


def ofdm_modulation(complex_numbers):
    """ j) Divide the complex number string into groups of 48 complex numbers. Each such group is
        associated with one OFDM symbol. In each group, the complex numbers are numbered 0 to 47 and
        mapped hereafter into OFDM subcarriers numbered -26 to -22, -20 to -8, -6 to -1, 1 to 6, 8 to 20,
        and 22 to 26. The subcarriers -21, -7, 7, and 21 are skipped and, subsequently, used for inserting
        pilot subcarriers. The 0 subcarrier, associated with center frequency, is omitted and filled with the
        value 0. Refer to 17.3.5.10 for details.
    """
    ofdm_symbols = np.reshape(complex_numbers, (-1, 48))
    ofdm_symbols = [map_to_carriers(symbol) for symbol in ofdm_symbols]
    # for symbol in symbols:


def test_2222():
    # IEEE Std 802.11-2016 - Table I-19—Interleaved bits of first DATA symbol
    input = '0111011111110000111011111100010001110011000000001011111100010001000100001001101000011101000100100110111' \
            '00011100011110101011010010001101101101011100110000100001100000000000011011011001101101101 '
    symbols = mapper(input, bits_per_symbol=4)
    symbols = list(range(48))

    # Table I-25—Time domain representation of the DATA field: symbol 1of 6
    expected = [(-0.139 + 0.05j), (0.004 + 0.014j), (0.011 - 0.1j), (-0.097 - 0.02j), (0.062 + 0.081j),
                (0.124 + 0.139j), (0.104 - 0.015j), (0.173 - 0.14j), (-0.04 + 0.006j), (-0.133 + 0.009j),
                (-0.002 - 0.043j), (-0.047 + 0.092j), (-0.109 + 0.082j), (-0.024 + 0.01j), (0.096 + 0.019j),
                (0.019 - 0.023j), (-0.087 - 0.049j), (0.002 + 0.058j), (-0.021 + 0.228j), (-0.103 + 0.023j),
                (-0.019 - 0.175j), (0.018 + 0.132j), (-0.071 + 0.16j), (-0.153 - 0.062j), (-0.107 + 0.028j),
                (0.055 + 0.14j), (0.07 + 0.103j), (-0.056 + 0.025j), (-0.043 + 0.002j), (0.016 - 0.118j),
                (0.026 - 0.071j), (0.033 + 0.177j), (0.02 - 0.021j), (0.035 - 0.088j), (-0.008 + 0.101j),
                (-0.035 - 0.01j), (0.065 + 0.03j), (0.092 - 0.034j), (0.032 - 0.123j), (-0.018 + 0.092j), -0.006j,
                (-0.006 - 0.056j), (-0.019 + 0.04j), (0.053 - 0.131j), (0.022 - 0.133j), (0.104 - 0.032j),
                (0.163 - 0.045j), (-0.105 - 0.03j), (-0.11 - 0.069j), (-0.008 - 0.092j), (-0.049 - 0.043j),
                (0.085 - 0.017j), (0.09 + 0.063j), (0.015 + 0.153j), (0.049 + 0.094j), (0.011 + 0.034j),
                (-0.012 + 0.012j), (-0.015 - 0.017j), (-0.061 + 0.031j), (-0.07 - 0.04j), (0.011 - 0.109j),
                (0.037 - 0.06j), (-0.003 - 0.178j), (-0.007 - 0.128j), (-0.059 + 0.1j), (0.004 + 0.014j),
                (0.011 - 0.1j), (-0.097 - 0.02j), (0.062 + 0.081j), (0.124 + 0.139j), (0.104 - 0.015j),
                (0.173 - 0.14j), (-0.04 + 0.006j), (-0.133 + 0.009j), (-0.002 - 0.043j), (-0.047 + 0.092j),
                (-0.109 + 0.082j), (-0.024 + 0.01j), (0.096 + 0.019j), (0.019 - 0.023j)]



    output = np.round(ofdm_modulation(symbols), 3)
    np.testing.assert_equal(expected[1:-1], output[
                                            1:-1])  # skipping first and last as they are involved in time windowing - which i will perform later


def test_inset_pilots_and_pad():
    input = [(-0.316 + 0.316j), (-0.316 + 0.316j), (0.316 + 0.316j), (-0.949 - 0.949j), (0.316 + 0.949j),
             (0.316 + 0.316j), (0.316 - 0.949j), (-0.316 - 0.949j), (-0.316 + 0.316j), (-0.949 + 0.316j),
             (-0.949 - 0.949j), (-0.949 - 0.949j), (0.949 + 0.316j), (0.316 + 0.316j), (-0.949 - 0.316j),
             (-0.949 - 0.316j), (-0.949 - 0.316j), (-0.949 - 0.949j), (0.949 - 0.316j), (0.949 + 0.949j),
             (-0.949 - 0.316j), (0.316 - 0.316j), (-0.949 - 0.316j), (-0.949 + 0.949j), (-0.316 + 0.949j),
             (0.316 + 0.949j), (-0.949 + 0.316j), (0.949 - 0.949j), (0.316 + 0.316j), (-0.316 - 0.316j),
             (-0.316 + 0.949j), (0.949 - 0.316j), (-0.949 - 0.316j), (0.949 + 0.316j), (-0.316 + 0.949j),
             (0.949 + 0.316j), (0.949 - 0.316j), (0.949 - 0.949j), (-0.316 - 0.949j), (-0.949 + 0.316j),
             (-0.949 - 0.949j), (-0.949 - 0.949j), (-0.949 - 0.949j), (0.316 - 0.316j), (0.949 + 0.316j),
             (-0.949 + 0.316j), (-0.316 + 0.949j), (0.316 - 0.316j)]

    expected = [0j, 0j, 0j, 0j, 0j, 0j, (-0.316 + 0.316j), (-0.316 + 0.316j), (0.316 + 0.316j), (-0.949 - 0.949j),
                (0.316 + 0.949j), (1 + 0j), (0.316 + 0.316j), (0.316 - 0.949j), (-0.316 - 0.949j), (-0.316 + 0.316j),
                (-0.949 + 0.316j), (-0.949 - 0.949j), (-0.949 - 0.949j), (0.949 + 0.316j), (0.316 + 0.316j),
                (-0.949 - 0.316j), (-0.949 - 0.316j), (-0.949 - 0.316j), (-0.949 - 0.949j), (1 + 0j), (0.949 - 0.316j),
                (0.949 + 0.949j), (-0.949 - 0.316j), (0.316 - 0.316j), (-0.949 - 0.316j), (-0.949 + 0.949j), 0j,
                (-0.316 + 0.949j), (0.316 + 0.949j), (-0.949 + 0.316j), (0.949 - 0.949j), (0.316 + 0.316j),
                (-0.316 - 0.316j), (1 + 0j), (-0.316 + 0.949j), (0.949 - 0.316j), (-0.949 - 0.316j), (0.949 + 0.316j),
                (-0.316 + 0.949j), (0.949 + 0.316j), (0.949 - 0.316j), (0.949 - 0.949j), (-0.316 - 0.949j),
                (-0.949 + 0.316j), (-0.949 - 0.949j), (-0.949 - 0.949j), (-0.949 - 0.949j), (-1 + 0j), (0.316 - 0.316j),
                (0.949 + 0.316j), (-0.949 + 0.316j), (-0.316 + 0.949j), (0.316 - 0.316j), 0j, 0j, 0j, 0j, 0j]

    output = insert_pilots_and_pad(input)
    np.testing.assert_equal(expected, np.round(output, 3))


def test_ofdm_i18():
    # IEEE Std 802.11-2016 - Table I-19—Interleaved bits of first DATA symbol
    input = '0111011111110000111011111100010001110011000000001011111100010001000100001001101000011101000100100110111' \
            '00011100011110101011010010001101101101011100110000100001100000000000011011011001101101101 '
    symbols = mapper(input, bits_per_symbol=4)
    symbols = list(range(48))

    # Table I-25—Time domain representation of the DATA field: symbol 1of 6
    expected = [(-0.139 + 0.05j), (0.004 + 0.014j), (0.011 - 0.1j), (-0.097 - 0.02j), (0.062 + 0.081j),
                (0.124 + 0.139j), (0.104 - 0.015j), (0.173 - 0.14j), (-0.04 + 0.006j), (-0.133 + 0.009j),
                (-0.002 - 0.043j), (-0.047 + 0.092j), (-0.109 + 0.082j), (-0.024 + 0.01j), (0.096 + 0.019j),
                (0.019 - 0.023j), (-0.087 - 0.049j), (0.002 + 0.058j), (-0.021 + 0.228j), (-0.103 + 0.023j),
                (-0.019 - 0.175j), (0.018 + 0.132j), (-0.071 + 0.16j), (-0.153 - 0.062j), (-0.107 + 0.028j),
                (0.055 + 0.14j), (0.07 + 0.103j), (-0.056 + 0.025j), (-0.043 + 0.002j), (0.016 - 0.118j),
                (0.026 - 0.071j), (0.033 + 0.177j), (0.02 - 0.021j), (0.035 - 0.088j), (-0.008 + 0.101j),
                (-0.035 - 0.01j), (0.065 + 0.03j), (0.092 - 0.034j), (0.032 - 0.123j), (-0.018 + 0.092j), -0.006j,
                (-0.006 - 0.056j), (-0.019 + 0.04j), (0.053 - 0.131j), (0.022 - 0.133j), (0.104 - 0.032j),
                (0.163 - 0.045j), (-0.105 - 0.03j), (-0.11 - 0.069j), (-0.008 - 0.092j), (-0.049 - 0.043j),
                (0.085 - 0.017j), (0.09 + 0.063j), (0.015 + 0.153j), (0.049 + 0.094j), (0.011 + 0.034j),
                (-0.012 + 0.012j), (-0.015 - 0.017j), (-0.061 + 0.031j), (-0.07 - 0.04j), (0.011 - 0.109j),
                (0.037 - 0.06j), (-0.003 - 0.178j), (-0.007 - 0.128j), (-0.059 + 0.1j), (0.004 + 0.014j),
                (0.011 - 0.1j), (-0.097 - 0.02j), (0.062 + 0.081j), (0.124 + 0.139j), (0.104 - 0.015j),
                (0.173 - 0.14j), (-0.04 + 0.006j), (-0.133 + 0.009j), (-0.002 - 0.043j), (-0.047 + 0.092j),
                (-0.109 + 0.082j), (-0.024 + 0.01j), (0.096 + 0.019j), (0.019 - 0.023j)]

    ll =  [(-0.05929270666092634 + 0.10042482428252697j), (0.004092425832804836 + 0.013971114172662676j),
                    (0.010903666720265978 - 0.10026173404728128j), (-0.0969928613000402 - 0.020345674659864317j),
                    (0.06212348476267961 + 0.08137998966200377j), (0.12359456583646128 + 0.13891680699519982j),
                    (0.10425641784570533 - 0.015055236061888765j), (0.17293533399100125 - 0.13979135676701132j),
                    (-0.03959342375162872 + 0.00585376374299371j), (-0.13348812707864993 + 0.008928852031717772j),
                    (-0.0015876881303042902 - 0.04328466219159789j), (-0.04727705896914973 + 0.09216561913195342j),
                    (-0.10901256091378983 + 0.08170544389728396j), (-0.023963291761805436 + 0.010407231593805355j),
                    (0.09649615798647845 + 0.018548964287522188j), (0.019109918968221166 - 0.022571512413802224j),
                    (-0.08733541518449783 - 0.04941058997064829j), (0.0023412045230000213 + 0.05812531798078045j),
                    (-0.021044059925451913 + 0.22845735790743554j), (-0.1028888512059156 + 0.022824496285043013j),
                    (-0.019259288592050172 - 0.17515420176297444j), (0.017824577673127465 + 0.13177369124242555j),
                    (-0.07101936861726499 + 0.16038017900700768j), (-0.15319016984862427 - 0.061925808817496826j),
                    (-0.10707274535876585 + 0.027885896924185815j), (0.055435027714656936 + 0.1400177000510419j),
                    (0.06991102941239553 + 0.10268353016004547j), (-0.05555789957182841 + 0.024901605645509597j),
                    (-0.04275667649982156 + 0.0016203528221623714j), (0.0156848407894347 - 0.118057880678692j),
                    (0.025543497650235992 - 0.07124844675819939j), (0.033279915751768294 + 0.17722910326209035j),
                    (0.019764235243201256 - 0.02136788424104452j), (0.0353311317137757 - 0.08842214149388344j),
                    (-0.008123782322331671 + 0.10069755609058978j), (-0.03490639006331869 - 0.009642457921076077j),
                    (0.06464854470948449 + 0.030423412412733365j), (0.09247339798685189 - 0.03388424128406053j),
                    (0.031943342474459936 - 0.12253468079475699j), (-0.017528910452331237 + 0.09169162842419701j),
                    (6.495233390363744e-05 - 0.005853762811671135j), (-0.006224736140928611 - 0.056102920532711695j),
                    (-0.019328015257496035 + 0.03975765242192204j), (0.053200043245204984 - 0.13137842249509707j),
                    (0.021769002473584376 - 0.13281153720186661j), (0.1041566202299465 - 0.031699964368789024j),
                    (0.16267106123268774 - 0.04451157509150934j), (-0.10485953346709179 - 0.02958701516138735j),
                    (-0.11030694376677275 - 0.06917482428252697j), (-0.007737892208550545 - 0.09188389756079952j),
                    (-0.04921514635239757 - 0.04282844025215299j), (0.08470256727603256 - 0.01723794529127162j),
                    (0.09012961900247923 + 0.06335079968823731j), (0.014880004444225936 + 0.15337564557883943j),
                    (0.04860941509026712 + 0.09380874699861658j), (0.011148050004280916 + 0.033977519977399735j),
                    (-0.011512668894409411 + 0.011642572630894122j), (-0.015241950200504291 - 0.017380478074182887j),
                    (-0.06057294884277535 + 0.03100651497813032j), (-0.07019683847371141 - 0.04034416539329226j),
                    (0.01141481975552918 - 0.10862814518848006j), (0.03706997198795421 - 0.059970946257163205j),
                    (-0.0032158076226736117 - 0.17750183725769234j), (-0.007205087226298508 - 0.12807950320208467j),
                    (-0.05929270666092634 + 0.10042482428252697j), (0.004092425832804836 + 0.013971114172662676j),
                    (0.010903666720265978 - 0.10026173404728128j), (-0.0969928613000402 - 0.020345674659864317j),
                    (0.06212348476267961 + 0.08137998966200377j), (0.12359456583646128 + 0.13891680699519982j),
                    (0.10425641784570533 - 0.015055236061888765j), (0.17293533399100125 - 0.13979135676701132j),
                    (-0.03959342375162872 + 0.00585376374299371j), (-0.13348812707864993 + 0.008928852031717772j),
                    (-0.0015876881303042902 - 0.04328466219159789j), (-0.04727705896914973 + 0.09216561913195342j),
                    (-0.10901256091378983 + 0.08170544389728396j), (-0.023963291761805436 + 0.010407231593805355j),
                    (0.09649615798647845 + 0.018548964287522188j), (0.019109918968221166 - 0.022571512413802224j)]


    output = np.round(ll, 3)
    np.testing.assert_equal(expected[1:-1], output[
                                            1:-1])  # skipping first and last as they are involved in time windowing - which i will perform later



    output = np.round(ifft_guard(symbols), 3)
    np.testing.assert_equal(expected[1:-1], output[
                                            1:-1])  # skipping first and last as they are involved in time windowing - which i will perform later
