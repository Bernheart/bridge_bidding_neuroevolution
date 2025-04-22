import csv

from batch import BatchDistributor, batch_from_file
from evolution_agent import EvoAgent
from neural_net import NeuralNet
from saving_population import save_population
from show_model_bidding import print_model_bidding
from utils_functions import evolve, evaluation_fitness_all


def run_evolution(population_size=200, num_generations=400, batch_distributor=BatchDistributor()):
    population = [EvoAgent(NeuralNet()) for _ in range(population_size)]
    stats = []

    for generation in range(num_generations):
        batch = batch_distributor.get_random_batch()
        # Evolve population
        population = evolve(population, batch)
        # Evaluate fitness for stats
        scores, lengths = evaluation_fitness_all(population, batch, for_stats=True)

        best_score = max(scores)
        avg_score = sum(scores) / len(scores)
        worst_score = min(scores)

        best_length = max(lengths)
        avg_length = sum(lengths) / len(lengths)
        worst_length = min(lengths)

        print(f"Gen {generation}: Best Score={best_score:.3f}, Avg Score={avg_score:.3f}, Worst Score={worst_score:.3f}")
        print(f"Best Length={best_length:.3f}, Avg Length={avg_length:.3f}, Worst Length={worst_length:.3f}")
        stats.append((generation, best_score, avg_score, worst_score, best_length, avg_length, worst_length))

        # Save stats to CSV
        with open("evolution_stats.csv", "w", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(["Generation", "Best_Score", "Average_Score", "Worst_Score",
                             "Best_Length", "Average_Length", "Worst_Length"])
            writer.writerows(stats)

        print_model_bidding(agent=population[0], batch=batch_from_file(360), n=3)

    return population  # Final evolved population


if __name__ == "__main__":
    evolved_population = run_evolution()
    save_population(evolved_population)
