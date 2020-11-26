import numpy as np
import torch

class Color:
    """
        Colors class for use with terminal.
    """
    HEADER = '\033[95m'
    OKBLUE = '\033[94m'
    OKGREEN = '\033[92m'
    WARNING = '\033[93m'
    FAIL = '\033[91m'
    ENDC = '\033[0m'
    BOLD = '\033[1m'
    UNDERLINE = '\033[4m'


class RunningMeanStd(object):
    """
    Class to handle running mean and standard deviation book-keeping.
    From https://github.com/openai/baselines/blob/1b092434fc51efcb25d6650e287f07634ada1e08/baselines/common/running_mean_std.py
    See https://en.wikipedia.org/wiki/Algorithms_for_calculating_variance#Parallel_algorithm
    """

    def __init__(self, epsilon=1e-4, shape=()):
        self.mean = np.zeros(shape, 'float64')
        self.var = np.ones(shape, 'float64')
        self.count = epsilon

    def update(self, x):

        if type(x) in [float, int]:
            batch_mean = x
            batch_var = 0
            batch_count = 1
        else:
            batch_mean = np.mean(x, axis=0)
            batch_var = np.var(x, axis=0)
            batch_count = x.shape[0]
        self.update_from_moments(batch_mean, batch_var, batch_count)

    def update_from_moments(self, batch_mean, batch_var, batch_count):
        self.mean, self.var, self.count = update_mean_var_count_from_moments(
            self.mean, self.var, self.count, batch_mean, batch_var, batch_count)

    def save_state(self):
        """
        Saves running statistics.
        """
        return tuple([self.mean, self.var, self.count])

    def restore_state(self, state):
        """
        Restores running statistics.
        """
        self.mean, self.var, self.count = state

def update_mean_var_count_from_moments(mean, var, count, batch_mean, batch_var, batch_count):
    """
    Calculate and return running mean and variance.
    """

    delta = batch_mean - mean
    tot_count = count + batch_count

    new_mean = mean + delta * batch_count / tot_count
    m_a = var * count
    m_b = batch_var * batch_count
    M2 = m_a + m_b + np.square(delta) * count * batch_count / tot_count
    new_var = M2 / tot_count
    new_count = tot_count

    return new_mean, new_var, new_count

def sample_action_from_logp(logp):
    """
        Returns integer [0..len(probs)-1] based on log probabilities.
        Log probabilities will be normalized.
    """
    # taken from https://github.com/openai/baselines/blob/master/baselines/common/distributions.py
    # this is trick to sample directly from log probabilties without exponentiation.
    u = np.random.uniform(size=np.shape(logp))
    return np.argmax(logp - np.log(-np.log(u)), axis=-1)

def explained_variance(ypred, y):
    """
    # from https://github.com/openai/random-network-distillation/blob/436c47b7d42ffe904373c7b8ab6b2d0cff9c80d8/utils.py
    Computes fraction of variance that ypred explains about y.
    Returns 1 - Var[y-ypred] / Var[y]
    interpretation:
        ev=0  =>  might as well have predicted zero
        ev=1  =>  perfect prediction
        ev<0  =>  worse than just predicting zero
    """

    assert y.ndim == 1 and ypred.ndim == 1

    vary = np.var(y)

    return -1 if vary == 0 else np.clip(1 - np.var(y-ypred)/vary, -1, 1)

def dump_data(X, name):
    """dumps a np / torch tensor to a file."""

    if type(X) is torch.Tensor:
        X = X.detach().cpu().numpy()

    with open(name+".txt", "wt") as f:
        f.write("min," + str(np.min(X))+"\n")
        f.write("max," + str(np.max(X))+"\n")
        f.write("mean," + str(np.mean(X))+"\n")
        f.write("std," + str(np.std(X))+"\n")
    np.savetxt(name+".csv", X, delimiter=",")


def default(x, default):
    """ Returns x if x is not none, otherwise default. """
    return x if x is not None else default

def validate_dims(x, dims, dtype=None):
    """ Makes sure x has the correct dims and dtype.
        None will ignore that dim.
    """

    if dtype is not None:
        assert x.dtype == dtype, "Invalid dtype, expected {} but found {}".format(str(dtype), str(x.dtype))

    assert len(x.shape) == len(dims), "Invalid dims, expected {} but found {}".format(dims, x.shape)

    assert all((a is None) or a == b for a,b in zip(dims, x.shape)), "Invalid dims, expected {} but found {}".format(dims, x.shape)