import os
import numpy as np

import matplotlib.pyplot as plt
import csv
import src.utils.globals as g
import time

import matplotlib

matplotlib.use('TkAgg')


def moving_average(data, window_size=5):
    if len(data) < window_size:
        return data
    return np.convolve(data, np.ones(window_size) / window_size, mode='valid')


def plot_stats(file_path=g.STATS_FILE_PATH):
    # turn on interactive mode
    plt.ion()

    # create two figures up front
    fig1, ax1 = plt.subplots()
    fig2, ax2 = plt.subplots()

    flag = True
    while flag:
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

        cut = 0
        window = 1  # You can adjust this

        # ---- Figure 1: Fitness ----
        ax1.clear()

        smoothed_gen = generations[cut + window - 1:]
        ax1.plot(smoothed_gen, moving_average(best_score[cut:], window), label="Best Score (smoothed)")
        ax1.plot(smoothed_gen, moving_average(avg_score[cut:], window), label="Average Score (smoothed)")
        ax1.plot(smoothed_gen, moving_average(worst_score[cut:], window), label="Worst Score (smoothed)")
        ax1.set_xlabel("Generation")
        ax1.set_ylabel("Fitness")
        ax1.set_title(f"Evolution Fitness Over Generations [{g.VERSION}]")
        ax1.legend()
        ax1.grid(True)

        # ---- Figure 2: Length ----
        ax2.clear()
        ax2.plot(smoothed_gen, moving_average(best_length[cut:], window), label="Best Length (smoothed)")
        ax2.plot(smoothed_gen, moving_average(avg_length[cut:], window), label="Average Length (smoothed)")
        ax2.plot(smoothed_gen, moving_average(worst_length[cut:], window), label="Worst Length (smoothed)")
        ax2.set_xlabel("Generation")
        ax2.set_ylabel("Length")
        ax2.set_title(f"Evolution Bid Length Over Generations [{g.VERSION}]")
        ax2.legend()
        ax2.grid(True)

        # redraw both figures
        fig1.canvas.draw()
        fig2.canvas.draw()

        if "OLD" in os.environ:
            flag = False
        else:
            # Draw and pause _within_ Matplotlib’s event loop
            plt.pause(60.0)  # updates figures and waits 60 seconds
    plt.ioff()
    plt.show()


if __name__ == "__main__":
    plot_stats()  # to change version -> change env VERSION
