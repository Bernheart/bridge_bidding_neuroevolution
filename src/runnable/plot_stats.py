import matplotlib.pyplot as plt
import csv
import src.utils.globals as g
import time

import matplotlib

matplotlib.use('TkAgg')


def plot_stats(file_path=g.STATS_FILE_PATH):
    # turn on interactive mode
    plt.ion()

    # create two figures up front
    fig1, ax1 = plt.subplots()
    fig2, ax2 = plt.subplots()

    while True:
        # storage lists
        generations = []
        best_score, avg_score, worst_score = [], [], []
        best_length, avg_length, worst_length = [], [], []

        # read CSV
        with open(file_path, "r") as f:
            reader = csv.DictReader(f)
            for row in reader:
                gen = int(row["Generation"])
                generations.append(gen)
                best_score.append(float(row["Best_Score"]))
                avg_score.append(float(row["Average_Score"]))
                worst_score.append(float(row["Worst_Score"]))
                best_length.append(float(row["Best_Length"]))
                avg_length.append(float(row["Average_Length"]))
                worst_length.append(float(row["Worst_Length"]))

        # ---- Figure 1: Fitness ----
        ax1.clear()
        ax1.plot(generations, best_score,  label="Best Score")
        ax1.plot(generations, avg_score,   label="Average Score")
        ax1.plot(generations, worst_score, label="Worst Score")
        ax1.set_xlabel("Generation")
        ax1.set_ylabel("Fitness")
        ax1.set_title("Evolution Fitness Over Generations")
        ax1.legend()
        ax1.grid(True)

        # ---- Figure 2: Length ----
        ax2.clear()
        ax2.plot(generations, best_length,  label="Best Length")
        ax2.plot(generations, avg_length,   label="Average Length")
        ax2.plot(generations, worst_length, label="Worst Length")
        ax2.set_xlabel("Generation")
        ax2.set_ylabel("Length")
        ax2.set_title("Evolution Bid Length Over Generations")
        ax2.legend()
        ax2.grid(True)

        # redraw both figures
        fig1.canvas.draw()
        fig2.canvas.draw()

        # Draw and pause _within_ Matplotlib’s event loop
        plt.pause(60.0)   # updates figures and waits 60 seconds


if __name__ == "__main__":
    plot_stats()  # to change version -> change env VERSION
