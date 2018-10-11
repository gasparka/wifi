from contextlib import suppress
import numpy as np


file = '/home/gaspar/Documents/tmp.csv'


def get_row4_complex(file, vertical=True, complex=True):
    import csv

    with open(file) as csvfile:
        reader = csv.reader(csvfile)
        data = [[xx.replace('–', '-') for xx in x] for x in reader]

    ret0 = []
    ret1 = []
    ret2 = []
    ret3 = []
    try:
        if vertical:
            for x in data[:]:
                ret0.append(float(x[1]) + float(x[2])*1j)
                ret1.append(float(x[4]) + float(x[5])*1j)
                ret2.append(float(x[7]) + float(x[8])*1j)
                ret3.append(float(x[10]) + float(x[11])*1j)
        else:
            for x in data[:]:
                ret0.append(float(x[1]) + float(x[2])*1j)
                ret0.append(float(x[4]) + float(x[5])*1j)
                ret0.append(float(x[7]) + float(x[8])*1j)
                ret0.append(float(x[10]) + float(x[11])*1j)
    except:
        pass

    ret = ret0 + ret1 + ret2 + ret3
    return ret


def get_row6_bit(file, vertical=True, complex=True):
    import csv

    with open(file) as csvfile:
        reader = csv.reader(csvfile)
        data = [[xx.replace('–', '-') for xx in x] for x in reader]

    ret0 = []
    ret1 = []
    ret2 = []
    ret3 = []
    ret4 = []
    ret5 = []
    try:
        if vertical:
            for x in data[:]:
                ret0.append(x[1])
                ret1.append(x[3])
                ret2.append(x[5])
                ret3.append(x[7])
                ret4.append(x[9])
                ret5.append(x[11])

        else:
            for x in data[:]:
                ret0.append(float(x[1]) + float(x[2])*1j)
                ret0.append(float(x[4]) + float(x[5])*1j)
                ret0.append(float(x[7]) + float(x[8])*1j)
                ret0.append(float(x[10]) + float(x[11])*1j)
    except:
        pass

    ret = ret0 + ret1 + ret2 + ret3 + ret4 + ret5
    ret = ''.join(str(x) for x in ret)
    return ret

ret = get_row6_bit(file)
# ret = get_row4_complex(file)
# ret = get_row4_complex(file, vertical=False)
pass
# i = p.to_float(p.trans_data[1], 12)
# q = p.to_float(p.trans_data[15], 12)
# iq = np.array(i + q*1j)
# save_complex64_file('/home/gaspar/git/pyha/pyha/common/data/from_tap.raw', iq)
# pass