from scipy.stats import wilcoxon


class Stats:
    @staticmethod
    def wilcoxon(x, y) -> tuple:
        res = wilcoxon(x=x, y=y)
        return res
