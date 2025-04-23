import matplotlib.pyplot as plt
import csv
import src.utils.globals as g

import matplotlib

matplotlib.use('TkAgg')


def plot_stats(file_path=g.STATS_FILE_PATH):

    generations, best, avg, worst = [], [], [], []

    with open(file_path, "r") as f:
        reader = csv.DictReader(f)
        for row in reader:
            generations.append(int(row["Generation"]))
            best.append(float(row["Best_Score"]))
            avg.append(float(row["Average_Score"]))
            worst.append(float(row["Worst_Score"]))

    plt.plot(generations, best, label="Best Score")
    plt.plot(generations, avg, label="Average Score")
    plt.plot(generations, worst, label="Worst Score")
    plt.xlabel("Generation")
    plt.ylabel("Fitness")
    plt.legend()
    plt.title("Evolution Fitness Over Generations")
    plt.grid(True)
    plt.show()


if __name__ == "__main__":
    plot_stats()  # to change version -> change env VERSION
