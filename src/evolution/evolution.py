import random

import numpy as np

from src.enviroment.batch import BatchDistributor, Batch
from src.evolution.evolution_agent import EvoAgent
from src.evolution.neural_net import NeuralNet
from src.utils.saving_files import save_population, create_version_directory, save_config, \
    add_version_to_changelog, save_stats
from src.utils.utils_functions import statistics, random_split_n
import src.utils.globals as g

MUTATION_RATE = g.EARLY_MUTATION_RATE
DIVERSITY_WEIGHT = g.EARLY_DIVERSITY_WEIGHT


def evolve(population: list[EvoAgent], batch: Batch, elitism=g.ELITISM_PROB, tournament_size=g.TOURNAMENT_SIZE,
           mutate_prob=g.MUTATION_PROB, crossover_prob=g.CROSSOVER_PROB):
    # 1. Evaluate all agents
    from src.enviroment.evaluation import evaluation_fitness_all
    fitness_scores = evaluation_fitness_all(population, batch)

    # 2. Sort and keep top performers
    fitness_scores.sort(key=lambda x: x[0], reverse=True)
    elite = [agent for _, agent in fitness_scores[:int(elitism * len(population))]]

    # 3. Reproduce
    offspring = []
    while len(elite) + len(offspring) < len(population):
        parents = []
        # Selecting two parents
        for i in range(2):
            contenders = random.sample(fitness_scores, tournament_size)
            parents.append(max(contenders, key=lambda x: x[0])[1])

        # get children
        for i in range(2):
            if np.random.random() < crossover_prob:
                child_model = parents[0].model.crossover(parents[1].model)
                child = EvoAgent(child_model)
            else:
                child = EvoAgent(parents[i].model.clone())

            if np.random.random() < mutate_prob:
                child = child.clone_and_mutate()

            offspring.append(child)

    evolved_population = elite + offspring
    while len(evolved_population) > len(population):
        evolved_population.pop()
    return evolved_population


def population_size(gen):
    # for gen in [1…200], linearly interpolate 300→600
    population_change = g.POPULATION_SIZE - g.EARLY_POPULATION_SIZE
    last_generation = g.EARLY_GENERATIONS[1]-1
    return int(g.EARLY_POPULATION_SIZE + population_change * gen / last_generation)


def mutation_rate(gen):
    mutation_rate_change = g.MUTATION_RATE - g.EARLY_MUTATION_RATE
    last_generation = g.EARLY_GENERATIONS[1]-1
    return float(g.EARLY_MUTATION_RATE + mutation_rate_change * gen / last_generation)


def run_evolution(batch_distributor=BatchDistributor()):
    global MUTATION_RATE, DIVERSITY_WEIGHT
    population = [EvoAgent(NeuralNet()) for _ in range(g.EARLY_POPULATION_SIZE)]

    # early phase
    for generation in range(g.EARLY_GENERATIONS[0], g.EARLY_GENERATIONS[1]):
        batch = batch_distributor.get_random_batch()
        # Evolve population
        population = evolve(population, batch)

        if generation == 1:
            # preparing archives
            create_version_directory()
            save_config()
            add_version_to_changelog()
            save_stats([], file_path=g.LOG_FILE_PATH)

        if generation % g.EVAL_INTERVAL == 0:
            statistics(generation, population, batch)

        # to not lose all data when not going all through
        if generation % g.SAVE_INTERVAL == 0:
            save_population(population)

        # ramp population
        while len(population) < population_size(generation):
            population.append(EvoAgent(NeuralNet()))
        print(f"population size: {population_size(generation)}")

        # decay mutation rate
        MUTATION_RATE = mutation_rate(generation)
        print(f"mutation rate: {MUTATION_RATE}")

    # middle phase
    population = random_split_n(population, g.NO_ISLANDS)
    DIVERSITY_WEIGHT = g.DIVERSITY_WEIGHT
    for generation in range(g.MIDDLE_GENERATIONS[0], g.MIDDLE_GENERATIONS[1]):
        batch = batch_distributor.get_random_batch()
        population_for_stats = []
        # Evolve population
        for island in range(g.NO_ISLANDS):
            population[island] = evolve(population[island], batch)
            population_for_stats += population[island]

        if generation % g.EVAL_INTERVAL == 0:
            statistics(generation, population_for_stats, batch)
            for island in range(g.NO_ISLANDS):
                statistics(generation, population[island], batch, island=island)

        # to not lose all data when not going all through
        if generation % g.SAVE_INTERVAL == 0:
            save_population(population_for_stats)

        # island migration
        if generation != g.MIDDLE_GENERATIONS[0] and generation % g.ISLAND_MIGRATION_EVERY == 0:
            island_population = len(population[0])
            for island in range(g.NO_ISLANDS):
                top = population[island][:island_population * g.ISLAND_MIGRATION_TOP]
                population[(island + 1) % g.NO_ISLANDS][-island_population:] = top

    # late phase
    all_population = []
    for island in range(g.NO_ISLANDS):
        all_population += population[island]
    population = all_population

    MUTATION_RATE = g.LATE_MUTATION_RATE
    DIVERSITY_WEIGHT = g.LATE_DIVERSITY_WEIGHT
    for generation in range(g.LATE_GENERATIONS[0], g.LATE_GENERATIONS[1]):
        batch = batch_distributor.get_random_batch()
        # Evolve population
        population = evolve(population, batch)

        if generation % g.EVAL_INTERVAL == 0:
            statistics(generation, population, batch)

        # to not lose all data when not going all through
        if generation % g.SAVE_INTERVAL == 0:
            save_population(population)

    return population  # Final evolved population
