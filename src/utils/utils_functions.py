import random

import numpy as np
import src.utils.globals as g


stats = []
log = []
island_stats = []


def statistics(generation, population, batch, island=-1):
    # Evaluate fitness for stats
    from src.enviroment.evaluation import evaluation_fitness_all
    scores, lengths = evaluation_fitness_all(population, batch, for_stats=True)

    best_score = max(scores)
    avg_score = sum(scores) / len(scores)
    worst_score = min(scores)

    best_length = max(lengths)
    avg_length = sum(lengths) / len(lengths)
    worst_length = min(lengths)

    print(f"Gen {generation}: Best Score={best_score:.3f}, "
          f"Avg Score={avg_score:.3f}, Worst Score={worst_score:.3f}")
    print(f"Best Length={best_length:.3f}, Avg Length={avg_length:.3f}, Worst Length={worst_length:.3f}")
    stats_tuple = (generation, best_score, avg_score, worst_score, best_length, avg_length, worst_length)

    from src.utils.saving_files import save_stats
    file_path = ""
    if island != -1:
        island_stats.append(stats_tuple)
        file_path = g.ISLAND_FILE_PATH.format(island=island)
    else:
        stats.append(stats_tuple)
        file_path = g.STATS_FILE_PATH
        if len(stats) % g.LOG_INTERVAL == 0:
            log.append([sum(stat) / g.LOG_INTERVAL for stat in stats[-g.LOG_INTERVAL:]])
            save_stats(log, file_path=g.LOG_FILE_PATH)

    # Save stats to CSV
    save_stats(stats, file_path=file_path)

    from src.enviroment.batch import batch_from_file
    from src.runnable.show_model_bidding import print_model_bidding
    if island == -1:
        print_model_bidding(agent=population[0], batch=batch_from_file(360), n=3)


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


def random_split_n(lst, n):
    random.shuffle(lst)  # in-place O(N) Fisher–Yates shuffle :contentReference[oaicite:0]{index=0}
    k = len(lst) // n
    return [lst[i * k:(i + 1) * k] for i in range(n)]  # four O(k) slices, total O(N)


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
