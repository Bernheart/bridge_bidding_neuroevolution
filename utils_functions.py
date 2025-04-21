import numpy as np

from evolution_agent import EvoAgent
from neural_net import NeuralNet


def evaluation_fitness(agent, opponents, play_game_fn):
    fitness = 0
    for opponent in opponents:
        result = play_game_fn(agent, opponent)
        fitness += result
    return fitness


def initialize_population(size, input_size, hidden_size, output_size):
    return [
        EvoAgent(NeuralNet(input_size, hidden_size, output_size))
        for _ in range(size)
    ]


def evolve(population, play_game_fn, retain_top=0.2, mutate_prob=0.8):
    # 1. Evaluate all agents
    fitness_scores = []
    for agent in population:
        # Play vs N random agents excluding self
        opponents = np.random.choice([a for a in population if a != agent], size=3, replace=True)
        score = evaluation_fitness(agent, opponents, play_game_fn)
        fitness_scores.append(score)

    # 2. Sort and keep top performers
    fitness_scores.sort(key=lambda x: x[1], reverse=True)
    survivors = [agent for _, agent in fitness_scores[:retain_top * len(population)]]

    # 3. Reproduce
    children = []
    while len(survivors) + len(children) < len(population):
        parent: EvoAgent = np.random.choice(survivors)
        if np.random.random() < mutate_prob:
            children.append(parent.clone_and_mutate())
        else:
            children.append(parent)  # clone without mutation (rare)

    return survivors + children
