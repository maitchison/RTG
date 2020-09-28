import csv
import pickle
import matplotlib.pyplot as plt
import os
import numpy as np

def ema(X, gamma='auto'):
    if gamma == "auto":
        window_length = (len(X) ** 0.5)
        gamma = 1 - (1 / window_length)

    warm_up = min(max(1, len(X) // 20), 100)
    y = np.mean(X[:warm_up])
    Y = []
    for x in X:
        y = gamma * y + (1 - gamma) * x
        Y.append(y)
    return Y


def parse(s):
    try:
        x = int(s)
        return x
    except:
        return s

def plot_experiment(path, plots=(("score_red", "score_green", "score_blue"), ("game_length")), **kwargs):
    print("=" * 60)

    for y_axis in plots:
        plot_graph(path=f"run/{path}", y_axis=y_axis, **kwargs)


def load_data(path):

    # this format was a terraible idea, all columns should be single values, not lists with spaces inbetween
    types = [np.int] * 2 + [np.float] * 3 + ["U128"] * 9 + [np.float] * 2

    filename = os.path.join(path, 'env.0.csv')
    with open(filename, "r") as f:
        header = f.readline()
        cols = header.split(",")
    types = types[:len(cols)]  # remove columns that arn't there (compatability with older versions)

    csv_data = np.genfromtxt(filename, dtype=types, delimiter=",", names=True)

    data = {}
    for p1 in "RGB":
        for p2 in "RGB":
            data[f"{p1}v{p2}"] = []

    # add result fields
    for result in ["result_red", "result_blue", "result_timeout"]:
        data[result] = []

    for team in ["red", "green", "blue"]:
        data[f"shots_missed_{team}"] = []

    for column_name in csv_data.dtype.names:
        data[column_name] = csv_data[column_name]

    if "player_count" in data:
        player_count = data["player_count"][0]
    else:
        player_count = 2  # just guess...?

    x = []
    step_counter = 0
    for row in csv_data:
        x.append(step_counter / 1e6)
        # step_counter += sum(int(actions) for actions in row["stats_actions"].split(" "))
        step_counter += row["game_length"] * player_count
        # extract out the players it stats we need

        for i, vs in enumerate(vs_order):
            data[vs].append(int(row["stats_player_hit"].split(" ")[i]))

        # calculate shots missed by each team
        for i in range(3):
            shots_hit = np.sum(int(row["stats_player_hit"].split(" ")[i]) for i in range(i*3, i*3+3))
            shots_fired = int(row["stats_shots_fired"].split(" ")[i])
            shots_missed = shots_fired - shots_hit
            if i == 0:
                data["shots_missed_red"].append(shots_missed)
            if i == 1:
                data["shots_missed_green"].append(shots_missed)
            if i == 2:
                data["shots_missed_blue"].append(shots_missed)

        # generate result statistics (this should really be in stats... but infer it for now)
        data["result_red"].append(1 if row["score_red"] == 10 else 0)
        data["result_blue"].append(1 if row["score_blue"] == 10 else 0)
        data["result_timeout"].append(1 if row["game_length"] == 1000 else 0)

    data["x"] = x

    return data

def plot_graph(path, xlim=None, smooth='auto', y_axis=("score_red", "score_green", "score_blue")):
    color_map = {
        "score_red": "lightcoral",
        "score_green": "lightgreen",
        "score_blue": "lightsteelblue",

        "RvR": "lightcoral",
        "RvG": "lightgreen",
        "RvB": "lightsteelblue",

        "GvR": "lightcoral",
        "GvG": "lightgreen",
        "GvB": "lightsteelblue",

        "BvR": "lightcoral",
        "BvG": "lightgreen",
        "BvB": "lightsteelblue",

        "game_length": "gray",

        "result_timeout": "gray",
        "result_red": "lightcoral",
        "result_blue": "lightsteelblue",
    }

    label_map = {
        "score_red": "Red",
        "score_green": "Green",
        "score_blue": "Blue",
        "game_length": "Game Length",
    }

    y_units_map = {
        "score_red": "Score",
        "score_green": "Score",
        "score_blue": "Score",
        "game_length": "Steps",
    }

    y_units_map.update({vs: "Hits" for vs in vs_order})

    data = load_data(path)
    x = data["x"]

    plt.title(path)

    show_raw = True
    remove_blank_data = True

    # check and remove zeros
    if remove_blank_data:
        good_data = []
        for y_name in y_axis:
            if not (max(data[y_name]) == min(data[y_name]) == 0):
                good_data.append(y_name)
        y_axis = good_data
        if len(y_axis) == 0:
            # no data don't plot
            return

    if show_raw:
        for y_name in y_axis:
            plt.plot(
                x,
                ema(data[y_name], 0),
                c=color_map.get(y_name, "red"),
                alpha=0.03
            )
        for y_name in y_axis:
            plt.plot(
                x,
                ema(data[y_name], 0.8),
                c=color_map.get(y_name, "red"),
                alpha=0.1
            )

    for y_name in y_axis:
        plt.plot(
            x,
            ema(
                data[y_name], smooth),
            label=label_map.get(y_name, y_name),
            c=color_map.get(y_name, "red")
        )

    if xlim is not None:
        plt.xlim(0, xlim)

    plt.legend(loc="upper center", bbox_to_anchor=(0.5, 0.99))
    plt.grid()
    plt.xlabel("agent step (M)")
    plt.ylabel(y_units_map.get(y_axis[0], ""))
    plt.show()


vs_order = []
for p1 in "RGB":
    for p2 in "RGB":
        vs_order.append(f"{p1}v{p2}")