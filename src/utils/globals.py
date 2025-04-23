import yaml

SUITS = 5
LEVELS = 7
POSSIBLE_CONTRACTS = SUITS * LEVELS
POSSIBLE_BIDS = POSSIBLE_CONTRACTS + 1  # +pas
BIDDING_MASK_SIZE = POSSIBLE_BIDS + 1  # +last pas

CARD_IN_DECK = 52
HAND_MASK_SIZE = CARD_IN_DECK + 1  # +vul

CONFIG_FILE_PATH = "run_config.yaml"


def config():
    with open(CONFIG_FILE_PATH) as f:
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

INPUT_SIZE = cfg['neural_network']['input_size']
OUTPUT_SIZE = cfg['neural_network']['output_size']
HIDDEN_SIZE = cfg['neural_network']['hidden_layers'][0]

IMPS_LAMBDA = cfg['simulation']['scoring']['imp_difference']
SUIT_LAMBDA = cfg['simulation']['scoring']['good_suit_reward']
LENGTH_LAMBDA = cfg['simulation']['scoring']['bidding_length_bonus']
ENTROPY_LAMBDA = cfg['simulation']['scoring']['diversity_bonus']

# LOG_INTERVAL = cfg['logging']['log_interval']
EVAL_INTERVAL = cfg['logging']['eval_interval']
SAVE_INTERVAL = cfg['logging']['save_interval']

BATCH_SIZE = cfg['data']['batch']['size']
FIRST_BATCH = cfg['data']['batch']['first']
LAST_BATCH = cfg['data']['batch']['last']

BATCH_FILE_PATH = f'./generator/data/{BATCH_SIZE}/batch{{batch_number}}.csv'
NO_BATCHES_FILE_PATH = f'./generator/data/no_batches{BATCH_SIZE}.txt'

POPULATION_FILE_PATH = RESULTS_DIR + cfg['data']['files']['population']
STATS_FILE_PATH = RESULTS_DIR + cfg['data']['files']['statistics']
CONFIG_FILE_PATH_TO_COPY = RESULTS_DIR + CONFIG_FILE_PATH
CHANGELOG_FILE_PATH = cfg['data']['files']['changelog']
