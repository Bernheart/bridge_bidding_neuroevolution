import numpy as np

from batch import Batch
from brigde import point_diff_to_imps
from evolution_agent import EvoAgent
from neural_net import NeuralNet
import globals as g

# lambda_entropy = 0.01  # Controls influence of diversity
# diversity_bonus = behavior_diversity(agent, population)
#
# # Normalize by max diversity in the population
# max_div = max(1e-6, max(behavior_diversity(a, population) for a in population))
# normalized_bonus = diversity_bonus / max_div
#
# # Final fitness
# score = game_score + lambda_entropy * normalized_bonus

def initialize_population(size, input_size, hidden_size, output_size):
    return [
        EvoAgent(NeuralNet(input_size, hidden_size, output_size))
        for _ in range(size)
    ]


def behavior_diversity(agent, population):
    # A simple proxy: average distance between agent’s weights and others'
    distances = []
    for other in population:
        if other == agent: continue
        dist = np.linalg.norm(agent.model.w1 - other.model.w1)  # Only w1 for simplicity
        distances.append(dist)
    return np.mean(distances)


def normalize(list_to_normalize, eps=1e-8):
    arr = np.array(list_to_normalize, dtype=np.float32)
    min_val = arr.min()
    max_val = arr.max()
    # denominator is at least eps
    de_nom = (max_val - min_val) + eps
    return (arr - min_val) / de_nom


def evaluation_fitness(agent: EvoAgent, batch: Batch, for_show=False):
    imp_score = 0
    length = 0

    batch_size = len(batch)

    bidding_masks = []

    # looping through deals in batch
    for deal_index in range(batch_size):
        # creating all necessary preparations
        bidding_mask = [0 for _ in range(g.BIDDING_MASK_SIZE)]  # No bids at the start
        availability_mask = [1 for _ in range(g.BIDDING_MASK_SIZE)]
        availability_mask[len(availability_mask) - 1] = 0  # Cant have second pass imminently

        first_to_take_suit = [-1 for _ in range(g.SUITS)]
        last_bid_idx = 0
        # print(batch.hands[deal_index])

        bidding_length = 0
        # bidding loop
        while bidding_mask[len(bidding_mask) - 1] != 1:
            input_vector = batch.hands[deal_index][bidding_length % 2] + bidding_mask
            # print(batch.hands[deal_index][bidding_length % 2])
            # print(bidding_mask)
            # print(input_vector)
            output_vector = agent.model.forward(input_vector, availability_mask)
            bid_idx = int(np.argmax(output_vector))
            if bid_idx != len(output_vector) - 1:
                last_bid_idx = bid_idx
            bidding_mask[bid_idx] += 1
            availability_mask = [0 if i <= bid_idx else 1 for i in range(len(availability_mask))]
            suit_id = int((bid_idx - 1) / g.LEVELS)
            # print(len(output_vector))
            # print(bid_idx)
            # print(suit_id)
            # print()
            if suit_id != 5 and first_to_take_suit[suit_id] == -1:
                first_to_take_suit[suit_id] = bidding_length % 2
            bidding_length += 1
        length += bidding_length
        suit_id = int((last_bid_idx - 1) / g.LEVELS)
        score = batch.dd_tables[deal_index][first_to_take_suit[suit_id]][last_bid_idx]
        best_score = batch.best_score_for_deal(deal_index)
        imps = point_diff_to_imps(best_score - score)
        # print(imps)
        imp_score -= imps  # subtracting diff between best score and score

        if for_show:
            bidding_masks.append(bidding_mask)

    if for_show:
        return bidding_masks

    return imp_score / batch_size, length / batch_size


def evaluation_fitness_all(population: list[EvoAgent], batch: Batch, for_stats=False):
    lambda_entropy = 0.1  # Controls influence of diversity
    lambda_length = 0.05

    diversities = []
    scores = []
    lengths = []

    for agent in population:
        diversity = behavior_diversity(agent, population)
        score, length = evaluation_fitness(agent, batch)

        diversities.append(diversity)
        scores.append(score)
        lengths.append(length)

    if for_stats:
        return scores, lengths

    diversities = normalize(diversities)
    scores = normalize(scores)
    lengths = normalize(lengths)

    fitness_scores = scores + diversities * lambda_entropy + lengths * lambda_length
    return list(zip(fitness_scores, population))


def evolve(population: list[EvoAgent], batch: Batch, retain_top=0.1, mutate_prob=0.9, crossover_prob=0.3):
    # 1. Evaluate all agents
    fitness_scores = evaluation_fitness_all(population, batch)

    # 2. Sort and keep top performers
    fitness_scores.sort(key=lambda x: x[0], reverse=True)
    survivors = [agent for _, agent in fitness_scores[:int(retain_top * len(population))]]

    # 3. Reproduce
    children = []
    while len(survivors) + len(children) < len(population):
        parent1 = np.random.choice(survivors)
        if np.random.random() < crossover_prob:
            parent2 = np.random.choice(survivors)
            child_model = parent1.model.crossover(parent2.model)
            child = EvoAgent(child_model)
        else:
            child = parent1.clone_and_mutate()

        if np.random.random() < mutate_prob:
            child = child.clone_and_mutate()

        children.append(child)

    return survivors + children
