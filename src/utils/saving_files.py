import csv
import gzip
import json
import os
import pickle
from datetime import date

import orjson

from src.evolution.evolution_agent import EvoAgent
from src.evolution.neural_net import NeuralNet
from src.utils import globals as g
import shutil


# def save_population(population):
#     # Save population
#     with open(g.POPULATION_FILE_PATH, "wb") as f:
#         pickle.dump(population, f)


def save_population(population: list[EvoAgent]):
    data = [agent.model.get_parameters() for agent in population]
    with gzip.open(g.JSON_POPULATION_FILE_PATH, "wb") as f:
        f.write(orjson.dumps(data))


def load_population():
    with gzip.open(g.JSON_POPULATION_FILE_PATH, 'rb') as f:
        loaded_params = orjson.loads(f.read())

    population = []
    for params in loaded_params:
        agent = EvoAgent(NeuralNet())
        agent.model.set_parameters(params)
        population.append(agent)
    return population


# def load_population(file_path=g.POPULATION_FILE_PATH):
#     # Load population
#     with open(file_path, "rb") as f:
#         population = pickle.load(f)
#     return population


def create_version_directory():
    dir_path = g.RESULTS_DIR
    os.makedirs(dir_path, exist_ok=g.OVERWRITE_VERSION)
    os.makedirs(g.RESULTS_DIR + 'islands', exist_ok=g.OVERWRITE_VERSION)


def save_config():
    # Basic copy: content + permissions
    src = g.CONFIG_FILE_PATH
    dst = g.CONFIG_FILE_PATH_TO_COPY
    shutil.copy(src, dst)
    # └─ copies file data and file permissions (mode bits)


def save_stats(stats, file_path=g.STATS_FILE_PATH):
    with open(file_path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["Generation", "Best_Score", "Average_Score", "Worst_Score",
                         "Best_Length", "Average_Length", "Worst_Length"])
        writer.writerows(stats)


def add_version_to_changelog():
    with open(g.CHANGELOG_FILE_PATH, 'r') as f:
        lines = f.readlines()

    last_empty = (lines and lines[-1].strip() == '')

    today = date.today()
    formatted_date = today.strftime("%d-%m-%Y")

    description = g.cfg["description"]
    with open(g.CHANGELOG_FILE_PATH, 'a') as f:
        if not last_empty:
            f.write('\n')
        f.write(f'## {g.VERSION} ({formatted_date})\n')
        f.write(f'- {description}')
