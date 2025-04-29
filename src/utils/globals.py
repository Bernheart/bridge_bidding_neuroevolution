import os

import yaml

ALL_POINTS = 40
CARDS_IN_HAND = 13
COLORS = 4
SUITS = 5
LEVELS = 7
POSSIBLE_CONTRACTS = 35
POSSIBLE_BIDS = 36  # possible contracts + pas
BIDDING_MASK_SIZE = 37  # possible bids + last pas

CARD_IN_DECK = 52
HAND_MASK_SIZE = CARD_IN_DECK + 1  # +vul

CONFIG_FILE_PATH = "run_config.yaml"
ARCHIVE_DIR_PATH = "./archive/"


def config():
    file_path = CONFIG_FILE_PATH
    if "VERSION" in os.environ:
        file_path = f'{ARCHIVE_DIR_PATH}v{os.environ["VERSION"]}/{file_path}'
    # print(file_path)
    with open(file_path) as f:
        _cfg = yaml.safe_load(f)
    return _cfg


cfg = config()

VERSION = cfg["version"]
RESULTS_DIR = cfg['results_dir'].format(version=VERSION)
OVERWRITE_VERSION = cfg['advanced']['overwrite_version']

POPULATION_SIZE = cfg['evolution']['population_size']
GENERATIONS = cfg['evolution']['generations']

CROSSOVER_PROB = cfg['evolution']['crossover']['probability']
MUTATION_PROB = cfg['evolution']['mutation']['probability']
MUTATION_RATE = cfg['evolution']['mutation']['rate']
MUTATION_SCALE = cfg['evolution']['mutation']['perturbation_scale']

ELITISM_PROB = cfg['evolution']['elitism']['keep_best']
TOURNAMENT_SIZE = cfg['evolution']['selection']['tournament_size']

INPUT_SIZE = cfg['neural_network']['input_size']
OUTPUT_SIZE = cfg['neural_network']['output_size']
HIDDEN_LAYERS = cfg['neural_network']['hidden_layers']
print(HIDDEN_LAYERS)

IMPS_POWER = cfg['simulation']['scoring']['imp_to_power']
IMPS_LAMBDA = cfg['simulation']['scoring']['imp_difference']
SUIT_LAMBDA = cfg['simulation']['scoring']['good_suit_reward']
LENGTH_LAMBDA = cfg['simulation']['scoring']['bidding_length_bonus']
SCORE_LAMBDA = cfg['simulation']['scoring']['better_than_pass_bonus']
# ENTROPY_LAMBDA = cfg['simulation']['scoring']['diversity_bonus']

LOG_INTERVAL = cfg['logging']['log_interval']
EVAL_INTERVAL = cfg['logging']['eval_interval']
SAVE_INTERVAL = cfg['logging']['save_interval']

BATCH_SIZE = cfg['data']['batch']['size']
FIRST_BATCH = cfg['data']['batch']['first']
LAST_BATCH = cfg['data']['batch']['last']
BATCHES_PER_GENERATION = cfg['data']['batch']['per_generation']

BATCH_FILE_PATH = f'./generator/data/{BATCH_SIZE}/batch{{batch_number}}.csv'
NO_BATCHES_FILE_PATH = f'./generator/data/no_batches{BATCH_SIZE}.txt'

POPULATION_FILE_PATH = RESULTS_DIR + cfg['data']['files']['population']
JSON_POPULATION_FILE_PATH = RESULTS_DIR + 'population.json.gz'
STATS_FILE_PATH = RESULTS_DIR + cfg['data']['files']['statistics']
CONFIG_FILE_PATH_TO_COPY = RESULTS_DIR + CONFIG_FILE_PATH
CHANGELOG_FILE_PATH = cfg['data']['files']['changelog']
LOG_FILE_PATH = RESULTS_DIR + cfg['data']['files']['log']
ISLAND_FILE_PATH = RESULTS_DIR + cfg['data']['files']['island']

ARCHIVE_SIZE = cfg['advanced']['archive_size']
DIVERSITY_WEIGHT = cfg['advanced']['diversity_weight']

phases = cfg['evolution']['phases']
EARLY_POPULATION_SIZE = phases['early']['population_size']
EARLY_GENERATIONS = phases['early']['generations']
EARLY_MUTATION_RATE = phases['early']['mutation_rate']
EARLY_DIVERSITY_WEIGHT = phases['early']['diversity_weight']

MIDDLE_GENERATIONS = phases['middle']['generations']
ISLAND_MIGRATION_EVERY = phases['middle']['island_migration']['every']
ISLAND_MIGRATION_TOP = phases['middle']['island_migration']['top']
NO_ISLANDS = phases['middle']['island_migration']['number']

LATE_GENERATIONS = phases['late']['generations']
LATE_MUTATION_RATE = phases['late']['mutation_rate']
LATE_DIVERSITY_WEIGHT = phases['late']['diversity_weight']
