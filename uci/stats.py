from dataclasses import dataclass, field
from typing import Sequence, TypeAlias, Union
import numpy as np
from scipy.stats import wilcoxon, friedmanchisquare
from sklearn.metrics import mean_absolute_error, mean_squared_error
from utils.constants import EXPERIMENT_VARIABLES_LABELS

# Types
Metric: TypeAlias = tuple[float, ...] | dict[str, float]
ArrayLike1D = Union[Sequence[float], np.ndarray]


@dataclass
class Wilcoxon:
    x: ArrayLike1D
    y: ArrayLike1D

    statistic: float = field(init=False)
    p_value: float = field(init=False)

    def test(self) -> None:
        res = wilcoxon(self.x, self.y)
        self.statistic, self.p_value = res[0], res[1]


@dataclass
class Friedman:
    samples: Sequence[Sequence[float]]

    statistic: float = 0.0
    p_value: float = 0.0

    def test(self) -> None:
        res = friedmanchisquare(self.samples)
        self.statistic, self.p_value = res[0], res[1]


@dataclass
class SimulationMetrics:
    true_data: np.ndarray
    simulation_data: np.ndarray

    coverage_percentage: Metric = None
    error_margin: Metric = None
    kolmogorov_smirnov_result: Metric = None
    anderson_darling_result: Metric = None

    def evaluate(
        self,
        confidence_level: float = 0.95,
        random_state: int | None = None,
        result_as_dict=False,
    ) -> None:
        np.random.seed(random_state)

        if not 0.80 <= confidence_level <= 0.95:
            print("NOTE: it's recommended to specify a confidence level on range 0.80 - 0.95")

        try:
            self.coverage_percentage = self.__calculate_coverage_percentage(confidence_level=confidence_level)
            self.error_margin = self.__calculate_error_margin(as_dict=result_as_dict)
            self.kolmogorov_smirnov_result = self.__kolmogorov_smirnov_test(as_dict=result_as_dict)
            self.anderson_darling_result = self.__anderson_darling_test(as_dict=result_as_dict)
        except Exception:
            import traceback

            traceback.print_exc()

        return None

    def __calculate_coverage_percentage(self, confidence_level: float = 0.95) -> dict[str, float]:
        """Computes the amount of patients that their confidence interval are in range for each experiment variable.

        Args:
            confidence_level (float, optional): Statistic confidence level. Defaults to 0.95.

        Returns:
            dict[str, float]: Dictionary with coverage percentages for each experiment variable
        """

        from scipy.stats import t

        # Ensure we work with numpy arrays (caller may pass pandas DataFrame)
        true_data = np.asarray(self.true_data)
        simulation_data = np.asarray(self.simulation_data)

        # simulation_data: 3Darray = [n_patients * n_replicates * n_variables]
        n_patients: int = simulation_data.shape[0]
        n_replicates: int = simulation_data.shape[1]
        n_variables: int = simulation_data.shape[2]

        degrees_freedom = n_replicates - 1
        t_value = t.ppf(1 - (1 - confidence_level) / 2, degrees_freedom)

        coverage_percentages = {}

        for var_idx in range(n_variables):
            confidence_intervals = []
            coverage_count = 0

            for patient_idx in range(n_patients):
                # Defining Confidence Interval from the Simulated Data for this variable
                mean = np.mean(simulation_data[patient_idx, :, var_idx])
                std_error = np.std(simulation_data[patient_idx, :, var_idx], ddof=1) / np.sqrt(n_replicates)
                margin_error = t_value * std_error

                ci_lower = mean - margin_error
                ci_upper = mean + margin_error

                confidence_intervals.append((ci_lower, ci_upper))

                # Checking if true value is inside calculated Confidence Interval
                # Coerce true_data into a (n_patients, n_variables) numpy array robustly
                td = np.asarray(self.true_data)
                if td.ndim == 1:
                    if td.size == n_patients * n_variables:
                        td = td.reshape((n_patients, n_variables))
                    else:
                        # best-effort: repeat/trim to match shape
                        td = np.resize(td, (n_patients, n_variables))
                elif td.ndim == 0:
                    td = np.full((n_patients, n_variables), float(td))

                true_val = float(td[patient_idx, var_idx])

                if ci_lower <= true_val <= ci_upper:
                    coverage_count += 1

            coverage_percentage = (coverage_count / n_patients) * 100

            # Use actual variable names from constants, fallback to generic name if index out of range
            if var_idx < len(EXPERIMENT_VARIABLES_LABELS):
                var_name = EXPERIMENT_VARIABLES_LABELS[var_idx]
            else:
                var_name = f"variable_{var_idx}"
            coverage_percentages[var_name] = coverage_percentage

        return coverage_percentages

    def __calculate_error_margin(self, as_dict=False) -> Metric:
        # Ensure numpy arrays for arithmetic operations
        true_data = np.asarray(self.true_data)
        simulation_data = np.asarray(self.simulation_data)

        # simulation_data: (n_patients, n_simulations, experiment_variables(5))
        simulation_mean = np.mean(simulation_data, axis=1)

        rmse = np.sqrt(mean_squared_error(true_data, simulation_mean))
        mae = mean_absolute_error(true_data, simulation_mean)
        mape = np.mean(np.abs((true_data - simulation_mean) / true_data)) * 100

        print(f"RMSE: {rmse:.2f}")
        print(f"MAE: {mae:.2f}")
        print(f"MAPE: {mape:.2f}%")

        if as_dict:
            return {
                "rmse": rmse,
                "mae": mae,
                "mape": mape,
            }
        return (rmse, mae, mape)

    def __kolmogorov_smirnov_test(self, as_dict=False) -> Metric:
        from scipy.stats import ks_2samp

        # Ensure numpy arrays
        true_data = np.asarray(self.true_data)
        simulation_data = np.asarray(self.simulation_data)

        # Handle different data shapes
        if len(simulation_data.shape) == 3:
            # For 3D simulation data, flatten all replicates for each variable
            # Compare true data with mean of simulated data across replicates
            sim_mean = np.mean(simulation_data, axis=1)  # Shape: (n_patients, n_variables)
            statistic, p_value = ks_2samp(true_data.flatten(), sim_mean.flatten())
        else:
            # For 1D data (used in individual tests)
            statistic, p_value = ks_2samp(true_data, simulation_data.flatten())

        print(f"Estadístico KS: {statistic:.4f}")
        print(f"Valor p de KS: {p_value:.4f}")

        if as_dict:
            return {
                "statistic": statistic,
                "p_value": p_value,
            }
        return statistic, p_value

    def __anderson_darling_test(self, as_dict=False) -> Metric:
        import scipy.stats as stats

        # Ensure numpy arrays
        true_data = np.asarray(self.true_data)
        simulation_data = np.asarray(self.simulation_data)

        # Handle different data shapes
        if len(simulation_data.shape) == 3:
            # For 3D simulation data, use mean across replicates for comparison
            sim_mean = np.mean(simulation_data, axis=1)  # Shape: (n_patients, n_variables)
            min_size = min(len(true_data.flatten()), len(sim_mean.flatten()))
            real_sample = np.random.choice(true_data.flatten(), min_size, replace=False)
            simulated_sample = np.random.choice(sim_mean.flatten(), min_size, replace=False)
        else:
            # For 1D data (used in individual tests)
            min_size = min(len(true_data), len(simulation_data))
            real_sample = np.random.choice(true_data, min_size, replace=False)
            simulated_sample = np.random.choice(simulation_data, min_size, replace=False)

        # Perform the Anderson-Darling test
        anderson_result = stats.anderson_ksamp([real_sample, simulated_sample])

        statistic = anderson_result.statistic
        significance_level = anderson_result.significance_level

        print(f"Anderson-Darling Statistic: {anderson_result.statistic:.4f}")
        print(f"Approximate Critical p-value: {anderson_result.significance_level:.3f}")

        if as_dict:
            return {
                "statistic": statistic,
                "significance_level": significance_level,
            }
        return (statistic, significance_level)


class StatsUtils:
    @staticmethod
    def confidenceinterval(mean, std, n, coef=0.95) -> tuple[np.ndarray[float], np.ndarray[float]]:
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

        from scipy.stats import norm

        sem = std / np.sqrt(n)
        conf_int = norm.interval(alpha=coef, loc=mean, scale=sem)
        arr_limite_inferior: np.ndarray[float] = conf_int[0]
        arr_limite_superior: np.ndarray[float] = conf_int[1]

        # print("conf_int: ", conf_int)
        # print("arr_l_inferior: ", arr_limite_inferior)
        # print("arr_l_superior: ", arr_limite_superior)

        return arr_limite_inferior, arr_limite_superior
