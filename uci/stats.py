from scipy.stats import wilcoxon, friedmanchisquare
from scipy import stats
import numpy as np


class Stats:
    @staticmethod
    def wilcoxon(x, y) -> tuple:
        res = wilcoxon(x=x, y=y)
        return res

    @staticmethod
    def friedman(*samples) -> tuple:
        res = friedmanchisquare(samples)
        return res

    @staticmethod
    def confidenceinterval(mean, std, n):
        conf_int = stats.norm.interval(0.95, loc=mean, scale=std/np.sqrt(n))
        return conf_int
