import numpy as np
from numpy import ndarray
import pandas as pd
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
    def confidenceinterval(
        mean, std, n, coef=0.95
    ) -> tuple[ndarray[float], ndarray[float]]:
        # Si la desviación estándar es 0, el intervalo de confianza es simplemente la media.
        if np.all(std == 0):
            return mean, mean

        sem = std / np.sqrt(n)
        conf_int = norm.interval(confidence=coef, loc=mean, scale=sem)
        arr_limite_inferior: ndarray[float] = conf_int[0]
        arr_limite_superior: ndarray[float] = conf_int[1]
        return arr_limite_inferior, arr_limite_superior

    @staticmethod
    def calibration_metric_predict(y_true, y_pred, intervals) -> int:
        """Devuelve la CANTIDAD de muestras cuyo valor verdadero cae dentro del intervalo
        de confianza definido por *alpha* para las probabilidades predichas.
        """

        within_interval = np.zeros(len(intervals), dtype=int)
        for i, alpha in enumerate(intervals):
            lower = np.percentile(y_pred, (1 - alpha) / 2 * 100, axis=0)
            upper = np.percentile(y_pred, (1 + alpha) / 2 * 100, axis=0)
            within_interval[i] = int(np.sum((y_true >= lower) & (y_true <= upper)))
        return within_interval

    @staticmethod
    def calibration_metric_simulation(y_true, lower, upper) -> int:
        match y_true:
            case float() | int():
                return int(lower <= y_true <= upper)
            case list() | np.ndarray() | pd.Series() | pd.DataFrame():
                y_true = np.array(y_true)
                within_interval = int(np.sum((y_true >= lower) & (y_true <= upper)))
                return within_interval
            case _:
                raise TypeError(f"Tipo no soportado para y_true: {type(y_true)}")
