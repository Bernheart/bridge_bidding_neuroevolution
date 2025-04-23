import os

from src.evolution.evolution import run_evolution
from src.utils.saving_files import save_population, save_config, create_version_directory, add_version_to_changelog
from src.utils.utils_functions import next_version
from src.utils import globals as g


def core():
    if not g.OVERWRITE_VERSION and os.path.isdir(g.RESULTS_DIR):
        print(f"Version {g.VERSION} already exists, change version in run_config.yaml "
              f"(for ex. [ {g.VERSION} -> {next_version(g.VERSION)} ])")
        return
    # preparing archives
    create_version_directory()
    save_config()
    add_version_to_changelog()

    evolved_population = run_evolution()
    save_population(evolved_population)


if __name__ == "__main__":
    core()  # remember to run it from the project dir bc there are relative file paths
