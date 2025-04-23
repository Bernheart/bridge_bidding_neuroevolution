import numpy as np


def normalize(list_to_normalize, eps=1e-8):
    arr = np.array(list_to_normalize, dtype=np.float32)
    min_val = arr.min()
    max_val = arr.max()
    # denominator is at least eps
    de_nom = (max_val - min_val) + eps
    return (arr - min_val) / de_nom


def next_version(version: str) -> str:
    # assume the last char is a digit
    prefix, last = version[:-1], version[-1]
    new_last = str(int(last) + 1)
    return prefix + new_last


def inclusive_range(*args):
    """
    Usage:
      inclusive_range(n)           -> 1...n
      inclusive_range(start, stop) -> start...stop
      inclusive_range(start, stop, step)

    All endpoints inclusive.
    """
    # 1) Single integer -> range(1, n+1)
    if len(args) == 1 and isinstance(args[0], int):
        start, stop, step = 1, args[0], 1

    # 2) Sequence of length 2 or 3 -> unpack to ints
    elif len(args) == 1 and isinstance(args[0], (list, tuple)):
        seq = args[0]
        if not 2 <= len(seq) <= 3:
            raise TypeError("Sequence must be length 2 or 3")
        start, stop = seq[0], seq[1]
        step = seq[2] if len(seq) == 3 else 1

    # 3) Two or three separate ints
    elif 2 <= len(args) <= 3 and all(isinstance(x, int) for x in args):
        start, stop = args[0], args[1]
        step = args[2] if len(args) == 3 else 1

    else:
        raise TypeError(
            "inclusive_range expects:\n"
            "  • inclusive_range(n)\n"
            "  • inclusive_range(start, stop)\n"
            "  • inclusive_range(start, stop, step)\n"
            "  • or a single sequence [start, stop, (step)]"
        )

    # compute inclusive endpoint
    end = stop + (1 if step > 0 else -1)
    return range(start, end, step)
