import random

import numpy as np

from src.enviroment.batch import BatchDistributor, batch_from_file, Batch
from src.enviroment.evaluation import evaluation_fitness_all
from src.evolution.evolution_agent import EvoAgent
from src.evolution.neural_net import NeuralNet
from src.utils.saving_files import save_population, save_stats, create_version_directory, save_config, \
    add_version_to_changelog
from src.runnable.show_model_bidding import print_model_bidding
from src.utils.utils_functions import inclusive_range
import src.utils.globals as g


def evolve(population: list[EvoAgent], batch: Batch, elitism=g.ELITISM_PROB, tournament_size=g.TOURNAMENT_SIZE,
           mutate_prob=g.MUTATION_PROB, crossover_prob=g.CROSSOVER_PROB):
    # 1. Evaluate all agents
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


def run_evolution(population_size=g.POPULATION_SIZE, generations=g.GENERATIONS, batch_distributor=BatchDistributor()):
    population = [EvoAgent(NeuralNet()) for _ in range(population_size)]
    stats = []

    for generation in inclusive_range(generations):
        batch = batch_distributor.get_random_batch()
        # Evolve population
        population = evolve(population, batch)

        if generation == 1:
            # preparing archives
            create_version_directory()
            save_config()
            add_version_to_changelog()

        if generation % g.EVAL_INTERVAL == 0:
            # Evaluate fitness for stats
            scores, lengths = evaluation_fitness_all(population, batch, for_stats=True)

            best_score = max(scores)
            avg_score = sum(scores) / len(scores)
            worst_score = min(scores)

            best_length = max(lengths)
            avg_length = sum(lengths) / len(lengths)
            worst_length = min(lengths)

            print(f"Gen {generation}: Best Score={best_score:.3f}, "
                  f"Avg Score={avg_score:.3f}, Worst Score={worst_score:.3f}")
            print(f"Best Length={best_length:.3f}, Avg Length={avg_length:.3f}, Worst Length={worst_length:.3f}")
            stats.append((generation, best_score, avg_score, worst_score, best_length, avg_length, worst_length))

            # Save stats to CSV
            save_stats(stats)

            print_model_bidding(agent=population[0], batch=batch_from_file(360), n=3)

        # to not lose all data when not going all through
        if generation % g.SAVE_INTERVAL == 0:
            save_population(population)

    return population  # Final evolved population
