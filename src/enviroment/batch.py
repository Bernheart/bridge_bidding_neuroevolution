import csv
from random import randrange

import src.utils.globals as g
from src.utils.utils_functions import inclusive_range


def batch_from_file(batch_number: int):
    hands = []
    dd_tables = []

    file_path = g.BATCH_FILE_PATH.format(batch_number=batch_number)
    with open(file_path, 'r') as f:
        reader = list(csv.reader(f))
    rows = [[int(cell) for cell in row] for row in reader]  # Convert all cells to ints

    for row in range(0, len(rows), 4):
        if row + 3 >= len(rows):
            break  # Avoid going out of bounds if file ends unevenly
        hand_pair = (rows[row], rows[row + 1])
        dd_pair = (rows[row + 2], rows[row + 3])
        hands.append(hand_pair)
        dd_tables.append(dd_pair)

    return Batch(hands, dd_tables)


class Batch:
    def __init__(self, hands: list[tuple[list, list]], dd_tables: list[tuple[list, list]]):
        self.hands = hands
        self.dd_tables = dd_tables

    def __len__(self):
        return len(self.hands)

    def best_score_for_deal(self, deal_number: int) -> float:
        north_max = max(self.dd_tables[deal_number][0])
        south_max = min(self.dd_tables[deal_number][1])
        return max(north_max, south_max)

    def get_rewards_for_suit(self, deal_number: int) -> list[float]:
        from src.utils.utils_functions import normalize

        suits = []
        for suit in range(g.SUITS):
            suits.append(0)
            for level in range(g.LEVELS):
                for direction_table in self.dd_tables[deal_number]:
                    contract_id = 1 + suit * g.LEVELS + level
                    suits[suit] += direction_table[contract_id]
        return normalize(suits)


class BatchDistributor:
    def __init__(self):
        self.first_batch = g.FIRST_BATCH
        self.last_batch = g.LAST_BATCH
        self.batches = None
        self.get_batch_numbers()

    def __len__(self):
        return len(self.batches)

    def get_batch_numbers(self):
        if self.last_batch == -1:
            with open(g.NO_BATCHES_FILE_PATH, 'r') as file:
                self.last_batch = int(file.read().strip())
        self.batches = [i for i in inclusive_range(self.first_batch, self.last_batch)]

    def get_random_batch(self) -> Batch:
        batch_number = self.batches.pop(randrange(len(self.batches)))
        return batch_from_file(batch_number)
