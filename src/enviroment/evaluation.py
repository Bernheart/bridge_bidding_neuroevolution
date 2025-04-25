import numpy as np

from src.enviroment.batch import Batch
from src.enviroment.brigde import point_diff_to_imps
from src.evolution.evolution_agent import EvoAgent

import src.utils.globals as g
from src.utils.utils_functions import normalize


def behavior_diversity(agent, population):
    # A simple proxy: average distance between agent’s weights and others'
    distances = []
    for other in population:
        if other == agent:
            continue
        dist = np.linalg.norm(agent.model.w1 - other.model.w1)  # Only w1 for simplicity
        distances.append(dist)
    return np.mean(distances)


def evaluation_fitness(agent: EvoAgent, batch: Batch, for_show=False):
    imp_score = 0
    length = 0
    best_suit_rewards = 0

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
        hand_idx = bidding_length % 2
        # bidding loop
        while bidding_mask[len(bidding_mask) - 1] != 1:
            input_vector = [batch.points[deal_index][hand_idx]] + batch.colors[deal_index][hand_idx] + bidding_mask
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
                first_to_take_suit[suit_id] = hand_idx
            bidding_length += 1
        length += bidding_length
        suit_id = int((last_bid_idx - 1) / g.LEVELS)
        score = batch.dd_tables[deal_index][first_to_take_suit[suit_id]][last_bid_idx]
        best_score = batch.best_score_for_deal(deal_index)
        imps = point_diff_to_imps(best_score - score)
        # print(imps)
        imp_score -= imps ** 1.5  # subtracting diff between best score and score
        if last_bid_idx != 0:
            best_suit_rewards += batch.suit_rewards[deal_index][suit_id]
        else:
            best_suit_rewards += batch.suit_rewards[deal_index][g.SUITS]  # 5 index for pas

        if for_show:
            bidding_masks.append(bidding_mask)

    if for_show:
        return bidding_masks

    return imp_score / batch_size, length / batch_size, best_suit_rewards / batch_size


def evaluation_fitness_all(population: list[EvoAgent], batch: Batch, for_stats=False):
    diversities = []
    imps = []
    lengths = []
    suit_rewards = []

    for agent in population:
        diversity = behavior_diversity(agent, population)
        score, length, reward_for_good_suit = evaluation_fitness(agent, batch)

        diversities.append(diversity)
        imps.append(score)
        lengths.append(length)
        suit_rewards.append(reward_for_good_suit)

    if for_stats:
        return imps, lengths

    diversities = normalize(diversities)
    imps = normalize(imps)
    lengths = normalize(lengths)
    suit_rewards = normalize(suit_rewards)

    fitness_scores = (imps * g.IMPS_LAMBDA + diversities * g.ENTROPY_LAMBDA +
                      lengths * g.LENGTH_LAMBDA + suit_rewards * g.SUIT_LAMBDA)
    return list(zip(fitness_scores, population))
