import csv
from random import randrange

import numpy as np

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
        self.colors: list[tuple[list, list]] = []
        self.points: list[list[int]] = []
        self.prepare_colors()
        self.suit_rewards: list[list[float]] = \
            [self.get_rewards_for_suit(deal_number) for deal_number in range(len(self))]

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
            suits.append(-np.inf)
            for level in range(g.LEVELS):
                for direction_table in self.dd_tables[deal_number]:
                    contract_id = 1 + suit * g.LEVELS + level
                    if direction_table[contract_id] > suits[suit]:
                        suits[suit] = direction_table[contract_id]
        suits.append(0)  # 0 for playing pas
        return normalize(suits)

    def prepare_colors(self):
        for hands in self.hands:
            self.colors.append(([0, 0, 0, 0], [0, 0, 0, 0]))
            self.points.append([0, 0])
            for idx in range(len(hands)):
                for color in range(g.COLORS):
                    for card in range(g.CARDS_IN_HAND):
                        if hands[idx][color * g.CARDS_IN_HAND + card] == 1:
                            self.colors[-1][idx][color] += 1
                            self.points[-1][idx] += max(0, card - 8)
                for color in range(g.COLORS):
                    self.colors[-1][idx][color] /= g.CARDS_IN_HAND
                # print(hands[idx])
                # print(self.points[-1][idx])
                self.points[-1][idx] /= 40


class BatchDistributor:
    def __init__(self):
        self.first_batch = g.FIRST_BATCH
        if g.LAST_BATCH == -1:
            self.last_batch = self.first_batch + g.GENERATIONS * g.BATCHES_PER_GENERATION - 1
        else:
            self.last_batch = g.LAST_BATCH
        self.batches = None
        self.get_batch_numbers()

    def __len__(self):
        return len(self.batches)

    def get_batch_numbers(self):
        # if self.last_batch == -1:
        #     with open(g.NO_BATCHES_FILE_PATH, 'r') as file:
        #         self.last_batch = int(file.read().strip())
        self.batches = [i for i in inclusive_range(self.first_batch, self.last_batch)]

    def get_random_batch(self, no_batches=g.BATCHES_PER_GENERATION) -> Batch:
        batch = Batch([], [])
        for idx in range(no_batches):
            batch_number = self.batches.pop(randrange(len(self.batches)))
            batch = add_batches(batch, batch_from_file(batch_number))
        return batch


def add_batches(batch1: Batch, batch2: Batch) -> Batch:
    batch = Batch([], [])

    batch.hands = batch1.hands + batch2.hands
    batch.dd_tables = batch1.dd_tables + batch2.dd_tables
    batch.colors = batch1.colors + batch2.colors
    batch.points = batch1.points + batch2.points
    batch.suit_rewards = batch1.suit_rewards + batch2.suit_rewards

    return batch
