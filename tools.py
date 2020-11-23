import csv
import pickle
import matplotlib.colors
import matplotlib.pyplot as plt
import os
import numpy as np
from collections import defaultdict

def ema(X, gamma:float = -1):
    if gamma < 0:
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
        title = log_filename
        plot_graph(results, title=title, y_axis=y_axis, hold=True)
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
        title = path
        plot_graph(results, title=title, y_axis=y_axis, **kwargs)

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

    data = defaultdict(list)

    column_casts = {
        "epoch": float,
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

        infer_epoch = "epoch" not in column_names

        for line in f:
            row = line.split(",")

            for name, value, in zip(column_names, row):
                if name in column_casts:
                    value = column_casts[name](value)
                else:
                    value = str(value)
                data[name] += [value]

            if player_count is None:
                player_count = sum([int(x) for x in data["player_count"][0].split(" ")])

            step_counter += data["game_length"][-1] * player_count

            # convert the team stats to single columns
            for i, hit in enumerate(int(x) for x in str(data["stats_player_hit"][-1]).split(" ")):
                if vs_order[i] not in data:
                    data[vs_order[i]] = []
                data[vs_order[i]] += [hit]

            # convert the team stats to single columns
            for stat in ["deaths", "kills", "general_shot", "general_moved", "general_hidden", "tree_harvested"]:

                stats_name = f"stats_{stat}"

                if stats_name not in data:
                    continue

                for team, value in zip("RGB", (int(x) for x in str(data[stats_name][-1]).split(" "))):
                    field_name = f"{team}_{stat}"
                    data[field_name] += [value]

            if infer_epoch:
                data["epoch"].append(float(step_counter)/1e6)

            # make  epoch an into to group better
            data["epoch"][-1] = round(data["epoch"][-1], 1)

    return data

def plot_graph(data, title, xlim=None, y_axis=("score_red", "score_green", "score_blue"), hold=False):

    marking_map = {
        "GvG": "--",
        "RvR": "--",
        "BvB": "--"
    }

    color_map = {
        "score_red": "lightcoral",
        "score_green": "lightgreen",
        "score_blue": "lightsteelblue",

        "Rv": "lightcoral",
        "Gv": "lightgreen",
        "Bv": "lightsteelblue",

        "game_length": "gray"
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

    X = data["epoch"]

    plt.title(title)

    remove_blank_data = True

    # check names
    missing_column = False
    for y_name in y_axis:
        if y_name not in data:
            print(f"Warning, missing column {y_name}")
            missing_column = True
    if missing_column:
        return

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

    def group_it(X, Y):
        unique_x = sorted(set(X))
        x_map = {x: [] for x in unique_x}
        for x, y in zip(X, Y):
            x_map[x].append(y)
        X = unique_x
        Y = [np.mean(x_map[x]) for x in x_map]
        Y_err = [np.std(x_map[x])/np.sqrt(len(x_map[x])) for x in x_map]
        return (np.asarray(z) for z in (X, Y, Y_err))

    color_index = 0
    for y_name in y_axis:

        _X, _Y, _Y_err = group_it(X, data[y_name])

        for k, v in color_map.items():
            if y_name.startswith(k):
                col = v
                break
        else:
            col = plt.get_cmap("tab20")(color_index)[:3]
            color_index += 1

        if type(col) is str:
            col = matplotlib.colors.to_rgb(col)

        dark_col = tuple((c/4) for c in col)
        error = _Y_err*1.96

        # smoothing
        if len(_Y) > 1000:
            _Y = ema(_Y, 0.99)
            error = ema(error, 0.99)
        elif len(_Y) > 100:
            _Y = ema(_Y, 0.95)
            error = ema(error, 0.95)
        elif len(_Y) > 10:
            _Y = ema(_Y, 0.8)
            ema(error, 0.8)

        error = np.asarray(error)

        plt.fill_between(
            _X, _Y-error, _Y+error, alpha=0.10, facecolor=col, edgecolor=dark_col
        )

        line_style = marking_map.get(y_name,"-")

        plt.plot(
            _X,
            _Y,
            label=label_map.get(y_name, y_name),
            linestyle=line_style,
            c=col
        )

    if xlim is not None:
        plt.xlim(0, xlim)

    plt.legend(loc="upper center", bbox_to_anchor=(0.5, 0.99))
    plt.grid()
    plt.xlabel("agent epoch (Million steps)")
    plt.ylabel(y_units_map.get(y_axis[0], ""))
    if not hold:
        plt.show()


vs_order = []
for p1 in "RGB":
    for p2 in "RGB":
        vs_order.append(f"{p1}v{p2}")