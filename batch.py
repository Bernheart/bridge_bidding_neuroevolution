import csv
from random import randrange

batch_sizes = 64


def batch_from_file(batch_number: int):
    hands = []
    dd_tables = []

    with open(f'./generator/data/{batch_sizes}/batch{batch_number}.csv', 'r') as f:
        reader = list(csv.reader(f))
    rows = [[int(cell) for cell in row] for row in reader]  # Convert all cells to ints

    for row in range(0, len(rows), 4):
            if row + 3 >= len(rows):
                break  # Avoid going out of bounds if file ends unevenly
            hand_pair = (rows[row], rows[row+1])
            dd_pair = (rows[row+2], rows[row+3])
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


class BatchDistributor:
    def __init__(self):
        self.no_batches = None
        self.batches = None
        self.get_batch_numbers()

    def __len__(self):
        return len(self.batches)

    def get_batch_numbers(self):
        with open(f'./generator/data/no_batches{batch_sizes}.txt', 'r') as file:
            self.no_batches = int(file.read().strip())
        self.batches = [i for i in range(1, self.no_batches+1)]

    def get_random_batch(self) -> Batch:
        batch_number = self.batches.pop(randrange(len(self.batches)))
        return batch_from_file(batch_number)
