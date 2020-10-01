import csv
import pickle
import matplotlib.pyplot as plt
import os
import numpy as np
import matplotlib.pyplot as plt

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

LAST_PLOT_FILENAME = None

def export_graph(path, epoch=None):
    """
    Loads scores for experiment in given path and exports a PNG plot
    :param path:
    :return:
    """
    global LAST_PLOT_FILENAME
    results = load_results(path)
    y_axis = ("score_red", "score_green", "score_blue")
    plt.figure(figsize=(12,8)) # make it big
    plot_graph(results, path, y_axis=y_axis, hold=True)
    scores = tuple(round(get_score(results, team), 2) for team in ["red", "green", "blue"])
    end_tag = "" if epoch is None else f"[{epoch}]"
    filename = os.path.join(path, f"results {scores} {end_tag}.png")

    plt.savefig(filename, dpi=300)

    # clean up previous plot
    if LAST_PLOT_FILENAME is not None:
        os.remove(LAST_PLOT_FILENAME)

    LAST_PLOT_FILENAME = filename


def plot_experiment(path, plots=(("score_red", "score_green", "score_blue"), ("game_length",)), **kwargs):
    print("=" * 60)

    path = f"run/{path}"

    results = load_results(path)

    for y_axis in plots:
        plot_graph(results, path, y_axis=y_axis, **kwargs)

    scores = tuple(round(get_score(results, team), 2) for team in ["red", "green", "blue"])
    print(f"Scores {scores}")

def get_score(results, team, n_episodes=100):
    # get the score
    return np.mean(results[f"score_{team}"][-100:])

def get_score_alt(results, team, n_episodes=100):
    # get the score, which is a combination of the time taken to win and the score acheived
    return np.mean((results[f"score_{team}"] * 0.99 ** np.asarray(results["game_length"]))[-100:])

def load_results(path):

    # the idea here is to read in the csv files as a np table then convert it to a dictionary of columns

    filename = os.path.join(path, 'env_log.csv')

    types = [np.int] * 2 + [np.float] * 3 + ["U256"] * 10

    csv_data = np.genfromtxt(filename, delimiter=",", dtype=types, names=True)

    # create the hits stats
    data = {}
    for p1 in "RGB":
        for p2 in "RGB":
            data[f"{p1}v{p2}"] = []

    player_count = np.sum([int(x) for x in csv_data["player_count"][0].split(" ")])

    simple_columns = [
        "game_counter", "game_length", "score_red", "score_green", "score_blue"
    ]

    for column in simple_columns:
        data[column] = []

    x = []
    step_counter = 0
    for row in csv_data:
        x.append(step_counter / 1e6)
        # step_counter += sum(int(actions) for actions in row["stats_actions"].split(" "))
        step_counter += row["game_length"] * player_count

        # move data across
        for column in simple_columns:
            data[column].append(float(row[column]))

        # extract out the players it stats we need
        for i, vs in enumerate(vs_order):
            data[vs].append(int(row["stats_player_hit"].split(" ")[i]))

        # calculate shots missed by each team
        # for i in range(3):
        #     shots_hit = np.sum(int(row["stats_player_hit"].split(" ")[i]) for i in range(i*3, i*3+3))
        #     shots_fired = int(row["stats_shots_fired"].split(" ")[i])
        #     shots_missed = shots_fired - shots_hit
        #     if i == 0:
        #         data["shots_missed_red"].append(shots_missed)
        #     if i == 1:
        #         data["shots_missed_green"].append(shots_missed)
        #     if i == 2:
        #         data["shots_missed_blue"].append(shots_missed)

        # generate result statistics (this should really be in stats... but infer it for now)
        # data["result_red"].append(1 if row["score_red"] == 10 else 0)
        # data["result_blue"].append(1 if row["score_blue"] == 10 else 0)
        # data["result_timeout"].append(1 if row["game_length"] == 1000 else 0)

    data["x"] = x

    return data

def plot_graph(data, title, xlim=None, smooth='auto', y_axis=("score_red", "score_green", "score_blue"), hold=False):
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

    x = data["x"]

    plt.title(title)

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
    if not hold:
        plt.show()


vs_order = []
for p1 in "RGB":
    for p2 in "RGB":
        vs_order.append(f"{p1}v{p2}")