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
    def confidenceinterval(mean, std, n, coef=0.95) -> tuple[ndarray[float], ndarray[float]]:
        """
        Calcula el intervalo de confianza para una media dada, desviación estándar y tamaño de muestra.

        Args:
            mean (ndarray): Media de los datos.
            std (ndarray): Desviación estándar de los datos.
            n (int): Tamaño de la muestra.
            coef (float, opcional): Nivel de confianza (por defecto 0.95).

        Returns:
            tuple[ndarray, ndarray]: Límite inferior y superior del intervalo de confianza.
        """

        # Si la desviación estándar es 0, el intervalo de confianza es simplemente la media.
        if np.all(std == 0):
            return mean, mean

        sem = std / np.sqrt(n)
        conf_int = norm.interval(confidence=coef, loc=mean, scale=sem)
        arr_limite_inferior: ndarray[float] = conf_int[0]
        arr_limite_superior: ndarray[float] = conf_int[1]

        # print("conf_int: ", conf_int)
        # print("arr_l_inferior: ", arr_limite_inferior)
        # print("arr_l_superior: ", arr_limite_superior)

        return arr_limite_inferior, arr_limite_superior

    @staticmethod
    def calibration_metric_predict(y_true: ndarray, y_pred: ndarray, intervals: list[float]) -> ndarray[tuple[int]]:
        """
        Calcula la cantidad de valores verdaderos (`y_true`) que caen dentro de los intervalos de confianza
        definidos por los niveles especificados (`intervals`) para las probabilidades predichas (`y_pred`).

        Este método evalúa cuántos de los valores verdaderos están dentro de los intervalos de confianza
        generados para cada nivel en `intervals`. Los intervalos de confianza se calculan en base a los
        percentiles de las probabilidades predichas.

        Args:
            y_true (np.ndarray):
                Los valores verdaderos a evaluar. Debe ser un array de NumPy.
            y_pred (np.ndarray):
                Las probabilidades predichas. Debe ser un array de NumPy.
            intervals (list[float]):
                Una lista de niveles de confianza (por ejemplo, [0.8, 0.9, 0.95]) para calcular los intervalos.

        Returns:
            np.ndarray: Un array donde cada elemento representa la cantidad de valores verdaderos que caen dentro
            del intervalo de confianza correspondiente al nivel en `intervals`.

        Ejemplos:
            >>> y_true = np.array([0.1, 0.5, 0.9])
            >>> y_pred = np.array([0.2, 0.6, 0.8])
            >>> intervals = [0.8, 0.9]
            >>> StatsUtils.calibration_metric_predict(y_true, y_pred, intervals)
            array([2, 1])
        """

        within_interval = np.zeros(len(intervals), dtype=int)  # [0, 0, 0]

        for i, alpha in enumerate(intervals):
            lower = np.percentile(y_pred, (1 - alpha) / 2 * 100, axis=0)
            upper = np.percentile(y_pred, (1 + alpha) / 2 * 100, axis=0)
            within_interval[i] = int(np.sum((y_true >= lower) & (y_true <= upper)))

        return within_interval

    @staticmethod
    def calibration_metric_simulation(y_true: float | int | ndarray[float | int], lower: int, upper: int) -> int:
        """
        Calcula la cantidad de valores verdaderos (`y_true`) que caen dentro del intervalo especificado
        por los límites inferior (`lower`) y superior (`upper`).

        Este método soporta tanto valores individuales (e.g., un solo número) como colecciones de valores
        (e.g., listas, arrays de NumPy, Series o DataFrames de pandas). Determina cuántos de los valores
        proporcionados están dentro del intervalo [lower, upper].

        Args:
            y_true (float | int | list | np.ndarray | pd.Series | pd.DataFrame):
                Los valores verdaderos a verificar. Puede ser un único valor o una colección de valores.
            lower (float):
                El límite inferior del intervalo.
            upper (float):
                El límite superior del intervalo.

        Returns:
            int: La cantidad de valores en `y_true` que caen dentro del intervalo [lower, upper].

        Raises:
            TypeError: Si `y_true` no es un tipo soportado.

        Ejemplos:
            Para un único valor:

            >>> y_true = 3.5
            >>> lower = 2.0
            >>> upper = 4.0
            >>> StatsUtils.calibration_metric_simulation(y_true, lower, upper)
            1

            Para una colección de valores:

            >>> y_true = [1.5, 2.0, 3.5, 4.0]
            >>> lower = 2.0
            >>> upper = 4.0
            >>> StatsUtils.calibration_metric_simulation(y_true, lower, upper)
            3
        """

        match y_true:
            # De ser un solo valor, verificar su tipo.
            case float() | int():
                return int(lower <= y_true <= upper)

            # Verificar que es una colección de valores.
            case list() | np.ndarray() | pd.Series() | pd.DataFrame():
                y_true = np.array(y_true)
                within_interval = int(np.sum((y_true >= lower) & (y_true <= upper)))

                return within_interval

            case _:
                raise TypeError(f"Tipo no soportado para y_true: {type(y_true)}")
