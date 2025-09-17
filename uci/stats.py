import numpy as np
from numpy import ndarray
import pandas as pd
from scipy.stats import wilcoxon, friedmanchisquare, norm
from sklearn.metrics import mean_absolute_error, mean_squared_error


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
        Compute confidence interval for a given mean, standard deviation and sample size.

        Args:
            mean (ndarray): Mean of the data.
            std (ndarray): Standard deviation of the data.
            n (int): Sample size.
            coef (float, optional): Confidence level (default 0.95).

        Returns:
            tuple[ndarray, ndarray]: Lower and upper bounds of the confidence interval.
        """

        # If the standard deviation is zero, the confidence interval is the mean.
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
        Count how many true values (`y_true`) fall inside confidence intervals defined by levels in `intervals` for
        predicted probabilities (`y_pred`).

        This method evaluates how many true values lie within the confidence intervals generated for each level in
        `intervals`. Confidence intervals are computed from percentiles of the predicted probabilities.

        Args:
            y_true (np.ndarray): True values to evaluate. Must be a NumPy array.
            y_pred (np.ndarray): Predicted probabilities. Must be a NumPy array.
            intervals (list[float]): A list of confidence levels (e.g., [0.8, 0.9, 0.95]) to build intervals.

        Returns:
            np.ndarray: An array where each element is the count of true values that fall inside the respective
            confidence interval for the corresponding level in `intervals`.

        Examples:
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
        Count how many true values (`y_true`) fall inside the interval defined by `lower` and `upper`.

        This method supports both single scalar values (e.g., a single number) and collections (lists, NumPy arrays,
        pandas Series or DataFrames). It returns how many provided values fall inside [lower, upper].

        Args:
            y_true (float | int | list | np.ndarray | pd.Series | pd.DataFrame): True values to check. Can be a single
                value or a collection.
            lower (float): Lower bound of the interval.
            upper (float): Upper bound of the interval.

        Returns:
            int: The count of values in `y_true` that fall inside [lower, upper].

        Raises:
            TypeError: If `y_true` is not a supported type.

        Examples:
            For a single value:

            >>> y_true = 3.5
            >>> lower = 2.0
            >>> upper = 4.0
            >>> StatsUtils.calibration_metric_simulation(y_true, lower, upper)
            1

            For a collection of values:

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

            # Check if it's a collection of values.
            case list() | np.ndarray() | pd.Series() | pd.DataFrame():
                y_true = np.array(y_true)
                within_interval = int(np.sum((y_true >= lower) & (y_true <= upper)))

                return within_interval

            case _:
                raise TypeError(f"Unsupported type for y_true: {type(y_true)}")

    ##########################
    # MODEL CALIBRATION #
    ##########################

    @staticmethod
    def simulation_model_calibration(true_data: pd.DataFrame, predict_data: pd.DataFrame) -> None:
        pass

    def __calibration_metrics(
        n_patients: int,
        n_replics: int,
        full_replics: list,
        true_data: list,
        confidence_level: int = 0.95,
    ):
        if not 0.80 <= confidence_level <= 0.95:
            print("NOTE: it's recommended to have a confidence_level in the range 0.80 to 0.95")

        import scipy.stats as stats

        freedown_grades = n_replics - 1
        t_value = stats.t.ppf(1 - (1 - confidence_level) / 2, freedown_grades)
        confidence_intervals = []
        cobertura = 0

        # Checking which values are inside the confidence interval
        for i in range(n_patients):
            mean = np.mean(full_replics[i])
            std_error = np.std(full_replics[i], ddof=1) / np.sqrt(n_replics)
            margin_error = t_value * std_error

            ci_lower = mean - margin_error
            ci_upper = mean + margin_error

            confidence_intervals.append((ci_lower, ci_upper))

            if ci_lower <= true_data[i] <= ci_upper:
                cobertura += 1

        coverage_percentage = (cobertura / n_patients) * 100
        print(f"Confidence interval coverage percentage ({confidence_level * 100}%): {coverage_percentage:.2f}%")

        return coverage_percentage

    def error_metrics(real_data, prediction_means):
        rmse = np.sqrt(mean_squared_error(real_data, prediction_means))
        mae = mean_absolute_error(real_data, prediction_means)
        mape = np.mean(np.abs((real_data - prediction_means) / real_data)) * 100

        print(f"RMSE: {rmse:.2f}")
        print(f"MAE: {mae:.2f}")
        print(f"MAPE: {mape:.2f}%")

        return rmse, mae, mape

    def ks_test(true_data, complete_replics):
        from scipy.stats import ks_2samp

        datos_simulados_completos = complete_replics.flatten()
        ks_statistic, ks_pvalue = ks_2samp(true_data, datos_simulados_completos)

        print(f"KS statistic: {ks_statistic:.4f}")
        print(f"KS p-value: {ks_pvalue:.4f}")

        return ks_statistic, ks_pvalue

    def ad_test(true_sample, simulation_sample):
        from scipy.stats import anderson_ksamp

        anderson_result = anderson_ksamp([true_sample, simulation_sample])

        print(f"Anderson-Darling statistic: {anderson_result.statistic:.4f}")
        print(f"Approximate critical p-value: {anderson_result.significance_level:.3f}")

        return anderson_result

    ########################
    # ROBUSTNESS MEASUREMENT #
    ########################
    # TODO later
