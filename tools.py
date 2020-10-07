import csv
import pickle
import matplotlib.pyplot as plt
import os
import numpy as np

def ema(X, gamma:[str, float]='auto'):
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

LAST_PLOT_FILENAME = dict()

def export_graph(log_filename, epoch=None, png_base_name="results"):
    """
    Loads scores for experiment in given path and exports a PNG plot
    :param path:
    :return:
    """

    base_folder = os.path.split(log_filename)[0]

    try:
        results = load_results(log_filename)
        y_axis = ("score_red", "score_green", "score_blue")
        plt.figure(figsize=(12,8)) # make it big
        plot_graph(results, log_filename, y_axis=y_axis, hold=True)
        scores = tuple(round(float(get_score(results, team)), 2) for team in ["red", "green", "blue"])
        end_tag = "" if epoch is None else f"[{epoch}]"
        png_filename = os.path.join(base_folder, f"{png_base_name} {scores} {end_tag}.png")

        plt.savefig(png_filename, dpi=300)

        # clean up previous plot
        if log_filename in LAST_PLOT_FILENAME:
            os.remove(LAST_PLOT_FILENAME[log_filename])

        LAST_PLOT_FILENAME[log_filename] = png_filename

    finally:
        plt.close()


def plot_experiment(path, plots=(("score_red", "score_green", "score_blue"), ("game_length",)), **kwargs):
    print("=" * 60)

    path = f"run/{path}"

    results = load_results(path)

    for y_axis in plots:
        plt.figure(figsize=(12, 4))  # make it big
        plot_graph(results, path, y_axis=y_axis, **kwargs)

    scores = tuple(round(float(get_score(results, team)), 2) for team in ["red", "green", "blue"])
    print(f"Scores {scores}")

def get_score(results, team, n_episodes=100):
    # get the score
    return np.mean(results[f"score_{team}"][-100:])

def get_score_alt(results, team, n_episodes=100):
    # get the score, which is a combination of the time taken to win and the score acheived
    return np.mean((results[f"score_{team}"] * 0.99 ** np.asarray(results["game_length"]))[-100:])

def load_results(path):
    """
    return a dictionary of columns
    can't use genfromtex because of weird format for arrays that I used
    :param path:
    :return:
    """

    data = {
        'x': []
    }

    column_casts = {
        "env_name": str,
        "game_counter": int,
        "game_length": int,
        "score_red": float,
        "score_green": float,
        "score_blue": float,
        "wall_time": float,
        "date_time": float
    }

    # load in data
    step_counter = 0
    player_count = None
    with open(path, "r") as f:
        header = f.readline()
        column_names = [name.strip() for name in header.split(",")]
        for name in column_names:
            data[name] = []

        for line in f:
            row = line.split(",")

            for name, value, in zip(column_names, row):
                if name in column_casts:
                    value = column_casts[name](value)
                else:
                    value = str(value)
                data[name].append(value)

            if player_count is None:
                player_count = sum([int(x) for x in data["player_count"][0].split(" ")])

            step_counter += data["game_length"][-1] * player_count
            #step_counter += sum([int(x) for x in data["stats_actions"][0].split(" ")])

            data["x"].append(int(step_counter))

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