from more_itertools import flatten


def do(data, train_short, train_long, signal):
    parts = [train_short] + [train_long] + [signal] + data

    for i in range(1, len(parts)):
        parts[i][0] = (parts[i - 1][-64] + parts[i][0]) / 2

    result = list(flatten(parts))

    # smoother power on/off
    result[0] /= 2
    result.append(result[-64] / 2)

    return result
