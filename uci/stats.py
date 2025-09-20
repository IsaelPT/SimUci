from dataclasses import dataclass
from typing import TypeAlias
import numpy as np
from scipy.stats import wilcoxon, friedmanchisquare
from sklearn.metrics import mean_absolute_error, mean_squared_error
from utils.constants import EXPERIMENT_VARIABLES

# Types
_METRIC: TypeAlias = tuple[float, ...] | dict[str, float]


class Wilcoxon:
    def __init__(self):
        self.statistic: float = 0.0
        self.p_value: float = 0.0

    def test(self, x, y) -> None:
        res = wilcoxon(x, y)
        self.statistic = res[0]
        self.p_value = res[1]


class Friedman:
    def __init__(self):
        self.statistic: float = 0.0
        self.p_value: float = 0.0

    def test(self, *samples) -> None:
        res = friedmanchisquare(*samples)
        self.statistic = res[0]
        self.p_value = res[1]


@dataclass
class SimulationMetrics:
    true_data: np.ndarray
    simulation_data: np.ndarray

    coverage_percentage: _METRIC = None
    error_margin: _METRIC = None
    kolmogorov_smirnov_result: _METRIC = None
    anderson_darling_result: _METRIC = None

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

        true_data, simulation_data = self.true_data, self.simulation_data

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
                if ci_lower <= true_data[patient_idx, var_idx] <= ci_upper:
                    coverage_count += 1

            coverage_percentage = (coverage_count / n_patients) * 100
            # Use actual variable names from constants, fallback to generic name if index out of range
            if var_idx < len(EXPERIMENT_VARIABLES):
                var_name = EXPERIMENT_VARIABLES[var_idx]
            else:
                var_name = f"variable_{var_idx}"
            coverage_percentages[var_name] = coverage_percentage
            print(
                f"Porcentaje de cobertura de IC para {var_name} ({confidence_level * 100}%): {coverage_percentage:.2f}%"
            )

        return coverage_percentages

    def __calculate_error_margin(self, as_dict=False) -> _METRIC:
        true_data, simulation_data = self.true_data, self.simulation_data

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

    def __kolmogorov_smirnov_test(self, as_dict=False) -> _METRIC:
        from scipy.stats import ks_2samp

        true_data, simulation_data = self.true_data, self.simulation_data

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

    def __anderson_darling_test(self, as_dict=False) -> _METRIC:
        import scipy.stats as stats

        true_data, simulation_data = self.true_data, self.simulation_data

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
