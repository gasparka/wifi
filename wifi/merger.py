from hypothesis import given
from hypothesis._strategies import lists, complex_numbers
from more_itertools import flatten, chunked

from wifi import guard_interval


def do(train_short, train_long, frames):
    parts = [train_short] + [train_long] + frames

    for i in range(1, len(parts)):
        parts[i][0] = (parts[i - 1][-64] + parts[i][0]) / 2

    result = list(flatten(parts))

    # smoother power on/off
    result[0] /= 2
    result.append(result[-64] / 2)

    return result


# not a perfect reconstuction!
def undo(iq):
    short = iq[:160]
    long = iq[160:320]

    frame_size = guard_interval.SIZE + 64
    frames = list(chunked(iq[320:], frame_size))
    if len(frames[-1]) != frame_size:
        frames = frames[:-1]

    return short, long, frames


@given(lists(lists(complex_numbers(), min_size=80, max_size=80), min_size=1, max_size=32))
def test_hypothesis(frames):
    from wifi import preambler
    short = preambler.short_training_sequence()
    long = preambler.long_training_sequence()

    res_short, res_long, res_frames = undo(do(short, long, frames))

    # 'do' contaminates the first sample of each symbol, cannot be restored with 'undo'
    assert res_short[1:] == short[1:]
    assert res_long[1:] == long[1:]
    for frame, res_frame in zip(frames, res_frames):
        assert frame[1:] == res_frame[1:]
