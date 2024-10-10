import numpy as np
from scipy.stats import wilcoxon, friedmanchisquare, norm


class Wilcoxon:
    def __init__(self):
        self.statistic: float = 0
        self.p_value: float = 0

    def test(self, x, y) -> None:
        res = wilcoxon(x, y)
        self.statistic = res[0]
        self.p_value = res[1]


class Friedman:
    def __init__(self):
        self.statistic: float = 0
        self.p_value: float = 0

    def test(self, *samples) -> None:
        res = friedmanchisquare(*samples)
        self.statistic = res[0]
        self.p_value = res[1]


class StatsUtils:
    @staticmethod
    def confidenceinterval(mean, std, n):
        sem = std / np.sqrt(n)
        conf_int = norm.interval(confidence=0.95, loc=mean, scale=sem)
        arr_limite_inferior = conf_int[0]
        arr_limite_superior = conf_int[1]
        return arr_limite_inferior, arr_limite_superior
