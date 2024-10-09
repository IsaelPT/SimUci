import numpy as np
from scipy.stats import wilcoxon, friedmanchisquare, norm


class Wilcoxon:
    def __init__(self, x, y):
        self.sample1 = x,
        self.sample2 = y

    def wilcoxon_test(self) -> tuple:
        res = wilcoxon(self.sample1, self.sample2)
        return res


class Friedman:
    def __init__(self, *samples):
        self.samples = [*samples]

    def friedman_test(self) -> tuple[float, float]:
        res = friedmanchisquare(self.samples)
        return res


def confidenceinterval(mean, std, n):
    conf_int = norm.interval(0.95, loc=mean, scale=std / np.sqrt(n))
    return conf_int
