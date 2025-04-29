import numpy as np
import src.utils.globals as g
from src.enviroment.batch import batch_from_file
from src.utils.saving_files import load_population


def policy_entropy(probs):
    p = np.clip(probs, 1e-8, 1.0)
    return -np.sum(p * np.log(p))


def max_logit_gap(z, mask):
    # mask out invalid logits
    masked = np.where(mask, z, -np.inf)
    # find top two values
    top2 = np.partition(masked, -2)[-2:]
    return float(top2.max() - top2.min())


def compute_agent_stats(agent, batch):
    gaps = []
    entropies = []
    # run through each deal in batch, stepping through the bidding loop
    for deal_idx in range(len(batch)):
        bidding_mask = [0] * g.BIDDING_MASK_SIZE
        availability = [1] * g.BIDDING_MASK_SIZE
        availability[-1] = 0  # no immediate double‐pass

        bidding_length = 0
        # simulate bidding
        while bidding_mask[-1] != 1:
            # build input vector as you do in evaluation_fitness
            x = [batch.points[deal_idx][bidding_length % 2]] \
                + batch.colors[deal_idx][bidding_length % 2] \
                + bidding_mask

            # forward‐pass up to final‐logits
            z = np.dot(agent.model.wn[0], x) + agent.model.bn[0]
            a = np.tanh(z)
            for W, b in zip(agent.model.wn[1:-1], agent.model.bn[1:-1]):
                z = W.dot(a) + b
                a = np.tanh(z)
            z = agent.model.wn[-1].dot(a) + agent.model.bn[-1]

            # record stats
            gaps.append(max_logit_gap(z, availability))
            probs = np.exp((z - np.max(z)) * 1.0)  # no temp here
            probs = np.where(availability, probs, 0.0)
            probs /= probs.sum()
            entropies.append(policy_entropy(probs))

            # pick a bid (you can just step by argmax here)
            bid = int(np.argmax(probs))
            bidding_mask[bid] += 1
            availability = [0 if i <= bid else 1 for i in range(len(availability))]
            bidding_length += 1

    # return per‐agent means
    return np.mean(gaps), np.mean(entropies)


def monitor_population(population, batch):
    all_gaps, all_ent = [], []
    for agent in population:
        gap, ent = compute_agent_stats(agent, batch)
        all_gaps.append(gap)
        all_ent.append(ent)
    print(f"\n=== Gen x stats ===")
    print(f" Avg logit gap:     {np.mean(all_gaps):.3f}")
    print(f" Avg policy entropy:{np.mean(all_ent):.3f}\n")


if __name__ == "__main__":
    population = load_population()  # to change version -> change env VERSION

    batch_to_test = batch_from_file(2070)
    monitor_population(population, batch_to_test)
