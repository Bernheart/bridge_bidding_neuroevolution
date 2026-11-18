"""Action space and naming. Follows Pgx / OpenSpiel so observations and baselines are interchangeable.

Actions (38):  0 = Pass, 1 = Double (X), 2 = Redouble (XX), 3 + b = bid b,
with b = 5 * (level - 1) + strain and strains C, D, H, S, NT = 0..4 (bridge order: 1C < 1D < ... < 7NT).
Seats: N, E, S, W = 0..3 clockwise. Team = seat % 2 (0 = NS, 1 = EW).
"""

NUM_ACTIONS = 38
PASS, DOUBLE, REDOUBLE = 0, 1, 2
BID_OFFSET = 3
NUM_BIDS = 35
NUM_STRAINS = 5
NUM_SEATS = 4
OBS_DIM = 480
# History part of the observation: 1 group of opening passes + 35 bids x (bid, X, XX), 4 relative seats each.
HIST_GROUPS = 1 + 3 * NUM_BIDS  # 106
HIST_DIM = 4 * HIST_GROUPS  # 424
HAND_OFFSET = 4 + HIST_DIM  # 428
MAX_CALLS = 319  # longest possible auction (Pgx limit)

SEATS = "NESW"
STRAINS = "CDHSN"
STRAIN_SYMBOLS = ("♣", "♦", "♥", "♠", "NT")
SUIT_SYMBOLS = ("♣", "♦", "♥", "♠")
RANKS = "23456789TJQKA"  # rank index 0..12


def bid_action(level: int, strain: int) -> int:
    return BID_OFFSET + 5 * (level - 1) + strain


def action_level(action: int) -> int:
    return (action - BID_OFFSET) // 5 + 1


def action_strain(action: int) -> int:
    return (action - BID_OFFSET) % 5


def action_to_str(action: int, symbols: bool = True) -> str:
    if action == PASS:
        return "Pass"
    if action == DOUBLE:
        return "X"
    if action == REDOUBLE:
        return "XX"
    strain = action_strain(action)
    name = STRAIN_SYMBOLS[strain] if symbols else ("NT" if strain == 4 else STRAINS[strain])
    return f"{action_level(action)}{name}"


def str_to_action(s: str) -> int:
    s = s.strip().upper().replace("♣", "C").replace("♦", "D").replace("♥", "H").replace("♠", "S")
    if s in ("P", "PASS"):
        return PASS
    if s in ("X", "DBL", "D"):
        return DOUBLE
    if s in ("XX", "RDBL", "R"):
        return REDOUBLE
    level = int(s[0])
    rest = s[1:]
    strain = 4 if rest in ("N", "NT") else "CDHS".index(rest)
    if not 1 <= level <= 7:
        raise ValueError(f"bad level in {s!r}")
    return bid_action(level, strain)
