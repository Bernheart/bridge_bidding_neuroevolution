import numpy as np

from src.enviroment.batch import Batch
from src.enviroment.brigde import point_diff_to_imps
from src.evolution.evolution_agent import EvoAgent

import src.utils.globals as g
from src.utils.utils_functions import normalize

# def behavior_diversity(agent, population):
#     # A simple proxy: average distance between agent’s weights and others'
#     distances = []
#     for other in population:
#         if other == agent:
#             continue
#         dist = np.linalg.norm(agent.model.wn[0] - other.model.wn[0])  # Only w1 for simplicity
#         distances.append(dist)
#     return np.mean(distances)

# this will hold a fixed‐size buffer of past behavior vectors
behavior_archive = []


# -----------------------------------------------------------------------------
# your existing behavior. diversity fn, maybe renamed to get_descriptor()
# -----------------------------------------------------------------------------
def get_behavior_descriptor(agent):
    # Here you could switch to any compact “signature” — e.g. final hidden‐state,
    # average bid‐vector, etc.  For now we keep your weight‐distance proxy.
    return agent.model.wn[0].ravel().copy()


# -----------------------------------------------------------------------------
# helper to compute novelty w.r.t. the archive
# -----------------------------------------------------------------------------
def novelty_score(descriptor, archive):
    if not archive:
        return 0.0
    # distance to closest archived behavior
    dists = [np.linalg.norm(descriptor - other) for other in archive]
    return float(min(dists))


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

            output_vector = agent.model.forward(input_vector, availability_mask)
            bid_idx = int(np.argmax(output_vector))
            if bid_idx != len(output_vector) - 1:
                last_bid_idx = bid_idx
            bidding_mask[bid_idx] += 1
            availability_mask = [0 if i <= bid_idx else 1 for i in range(len(availability_mask))]
            suit_id = int((bid_idx - 1) / g.LEVELS)

            if suit_id != 5 and first_to_take_suit[suit_id] == -1:
                first_to_take_suit[suit_id] = hand_idx
            bidding_length += 1

        # calculating rewards
        length += bidding_length
        suit_id = int((last_bid_idx - 1) / g.LEVELS)
        score = batch.dd_tables[deal_index][first_to_take_suit[suit_id]][last_bid_idx]
        best_score = batch.best_score_for_deal(deal_index)
        imps = point_diff_to_imps(best_score - score)
        # print(imps)
        imp_score -= imps ** g.IMPS_POWER  # subtracting diff between best score and score
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
    global behavior_archive

    diversities = []
    imps = []
    lengths = []
    suit_rewards = []

    for agent in population:
        score, length, reward_for_good_suit = evaluation_fitness(agent, batch)

        # diversity bonus
        desc = get_behavior_descriptor(agent)
        div_bonus = novelty_score(desc, behavior_archive)
        agent._behavior_descriptor = desc

        diversities.append(div_bonus)
        imps.append(score)
        lengths.append(length)
        suit_rewards.append(reward_for_good_suit)

    if for_stats:
        return imps, lengths

    # Now update your archive (keep only most novel N descriptors)
    # collect all new descriptors
    new_descs = [a.behavior_descriptor for a in population]
    behavior_archive.extend(new_descs)

    # prune archive to keep only the top‐X most novel items
    # e.g. sort by novelty vs rest of archive, take highest
    scored = [(novelty_score(d, behavior_archive), d) for d in behavior_archive]
    scored.sort(reverse=True, key=lambda x: x[0])
    behavior_archive = [d for _, d in scored[:g.ARCHIVE_SIZE]]

    diversities = normalize(diversities)
    imps = normalize(imps)
    lengths = normalize(lengths)
    suit_rewards = normalize(suit_rewards)

    fitness_scores = (imps * g.IMPS_LAMBDA + diversities * g.DIVERSITY_WEIGHT +
                      lengths * g.LENGTH_LAMBDA + suit_rewards * g.SUIT_LAMBDA)
    return list(zip(fitness_scores, population))
