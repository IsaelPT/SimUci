from scipy.stats import wilcoxon
class Stats:

    def wilcoxon(self, x, y):
        res = wilcoxon(x=x,y=y)
        return res