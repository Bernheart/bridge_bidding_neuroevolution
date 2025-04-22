import pickle


def save_population(population):
    # Save population
    with open("population.pkl", "wb") as f:
        pickle.dump(population, f)


def load_population():
    # Load population
    with open("population.pkl", "rb") as f:
        population = pickle.load(f)
    return population
