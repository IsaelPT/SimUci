from dataclasses import dataclass, field
from typing import Sequence, TypeAlias, Union

import numpy as np
from scipy.stats import friedmanchisquare, wilcoxon
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
        res = friedmanchisquare(*self.samples)
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
        """Runs an evaluation calculating a few metrics: coverage_percentage, error_margin, ks_test, ad_test. The test is used to validate the simulation model data output, clasing REAL_DATA vs. SIM_DATA.

        Args:
            confidence_level (float, optional): Confidence Level used for Confidence Interval, and later on the *coverage_percentage*. Defaults to 0.95.
            random_state (int | None, optional): Determines if the random number generator (rng) uses a seed. This determines if the randomization values are *reproducible* or not. Defaults to None.
            result_as_dict (bool, optional): The way each test return behaves. If True, functions will return a dictionary with the tests result and get stored in the SimulationMetric object. Defaults to False.

        Returns:
            None: No return values.
        """

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
        simulation_data = np.asarray(self.simulation_data)

        # Expect a 3D simulation array: (n_patients, n_replicates, n_variables)
        if simulation_data.ndim != 3:
            raise ValueError("simulation_data must be a 3D array of shape (n_patients, n_replicates, n_variables)")

        n_patients: int = simulation_data.shape[0]
        n_replicates: int = simulation_data.shape[1]
        n_variables: int = simulation_data.shape[2]

        # Coerce and validate/adjust true_data once (be permissive and deterministic)
        td = np.asarray(self.true_data)

        if td.ndim == 0:
            td = np.full((n_patients, n_variables), float(td))
        elif td.ndim == 1:
            # If flat vector matches all values, reshape
            if td.size == n_patients * n_variables:
                td = td.reshape((n_patients, n_variables))

            # One true value per variable -> broadcast across patients
            elif td.size == n_variables:
                td = np.tile(td.reshape((1, n_variables)), (n_patients, 1))

            # One true value per patient -> broadcast across variables
            elif td.size == n_patients:
                if n_variables == 1:
                    td = td.reshape((n_patients, 1))
                else:
                    print("Warning: true_data provided as length n_patients; tiling values across variables")
                    td = np.tile(td.reshape((n_patients, 1)), (1, n_variables))

            elif td.size > n_patients * n_variables:
                print("Warning: true_data longer than needed; trimming to match simulation size")
                td = td.ravel()[: n_patients * n_variables].reshape((n_patients, n_variables))

            # Shorter than expected: best-effort resize (repeats elements)
            else:
                print(
                    "Warning: true_data shorter than expected; resizing by repeating elements to match simulation shape"
                )
                td = np.resize(td, (n_patients, n_variables))
        elif td.ndim == 2:
            # If there are more rows/cols than needed, trim. If fewer rows, repeat rows.
            rows, cols = td.shape

            if rows < n_patients or cols < n_variables:
                print(
                    "Warning: true_data dimensions smaller than simulation; resizing by repeating elements to match shape"
                )
                td = np.resize(td, (n_patients, n_variables))
            else:
                # Trim extra rows/cols deterministically
                td = td[:n_patients, :n_variables]
        else:
            # Higher dimensions are unexpected: flatten and resize
            print("Warning: true_data has >2 dimensions; flattening and resizing to match simulation shape")
            td = np.resize(td.ravel(), (n_patients, n_variables))

        # Compute per-patient means and standard errors
        means = np.mean(simulation_data, axis=1)  # shape (n_patients, n_variables)

        coverage_percentages: dict[str, float] = {}

        # Handle case with too few replicates: degenerate CI (point estimate)
        if n_replicates < 2:
            print("Warning: fewer than 2 replicates; confidence intervals will be degenerate (no variance available).")
            lower = means.copy()
            upper = means.copy()
        else:
            degrees_freedom = n_replicates - 1
            t_value = t.ppf(1 - (1 - confidence_level) / 2, degrees_freedom)

            stds = np.std(simulation_data, axis=1, ddof=1)
            sems = stds / np.sqrt(n_replicates)
            margin_errors = t_value * sems

            lower = means - margin_errors
            upper = means + margin_errors

        # Compute boolean matrix of whether true values are inside CIs
        in_ci = (td >= lower) & (td <= upper)

        # Coverage per variable (mean over patients)
        coverage_array = in_ci.mean(axis=0) * 100

        for var_idx in range(n_variables):
            if var_idx < len(EXPERIMENT_VARIABLES_LABELS):
                var_name = EXPERIMENT_VARIABLES_LABELS[var_idx]
            else:
                var_name = f"variable_{var_idx}"
            coverage_percentages[var_name] = float(coverage_array[var_idx])

        return coverage_percentages

    def __calculate_error_margin(self, as_dict=False) -> Metric:
        # Ensure numpy arrays for arithmetic operations
        true_data = np.asarray(self.true_data)
        simulation_data = np.asarray(self.simulation_data)

        # simulation_data: (n_patients, n_simulations, experiment_variables(5))
        simulation_mean = np.mean(simulation_data, axis=1)

        # Make sure true_data has the same shape as simulation_mean for metric calculations
        td = np.asarray(true_data)
        if td.shape != simulation_mean.shape:
            # If total sizes match, try to reshape; otherwise resize (best-effort)
            if td.size == simulation_mean.size:
                try:
                    td = td.reshape(simulation_mean.shape)
                except Exception:
                    td = np.resize(td, simulation_mean.shape)
            else:
                td = np.resize(td, simulation_mean.shape)

        # Compute RMSE and MAE
        rmse = np.sqrt(mean_squared_error(td, simulation_mean))
        mae = mean_absolute_error(td, simulation_mean)

        # Compute MAPE safely: avoid division by zero by ignoring positions where true value is zero
        denom = td.astype(float)
        zero_mask = denom == 0
        if np.all(zero_mask):
            mape = float("nan")

            print("MAPE: cannot compute because all true values are zero (division by zero). Returning NaN.")
        else:
            # ignore entries where denom == 0
            with np.errstate(divide="ignore", invalid="ignore"):
                mape_vec = np.abs((td - simulation_mean) / denom)
                mape = np.nanmean(np.where(zero_mask, np.nan, mape_vec)) * 100

        print(f"RMSE: {rmse:.2f}")
        print(f"MAE: {mae:.2f}")
        print(f"MAPE: {mape if not np.isnan(mape) else 'nan'}%")

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
        if simulation_data.ndim == 3:
            # For 3D simulation data, compare per-variable distributions (flatten patients & replicates)
            n_vars = simulation_data.shape[2]
            per_var = []

            for v in range(n_vars):
                sim_flat = simulation_data[:, :, v].ravel()
                true_flat = true_data[:, v].ravel() if true_data.ndim > 1 else true_data.ravel()

                try:
                    stat, p = ks_2samp(true_flat, sim_flat)
                except Exception:
                    stat, p = float("nan"), float("nan")
                per_var.append((float(stat), float(p)))

            # Aggregate statistic (mean of statistics) for a quick overall indicator
            stats_arr = np.array([s for s, _ in per_var], dtype=float)
            pvals_arr = np.array([p for _, p in per_var], dtype=float)
            overall_stat = float(np.nanmean(stats_arr))
            overall_p = float(np.nanmean(pvals_arr))

            print(f"Estadístico KS (per-variable): {per_var}")
            print(f"Estadístico KS (overall mean): {overall_stat:.4f}")
            print(f"Valor p de KS (overall mean): {overall_p:.4f}")

            if as_dict:
                # Use human-readable variable labels when available, fallback to var_{i}
                per_variable_dict = {}
                for i, (s, p) in enumerate(per_var):
                    try:
                        var_name = (
                            EXPERIMENT_VARIABLES_LABELS[i] if i < len(EXPERIMENT_VARIABLES_LABELS) else f"var_{i}"
                        )
                    except Exception:
                        var_name = f"var_{i}"
                    per_variable_dict[var_name] = {"statistic": s, "p_value": p}

                return {
                    "per_variable": per_variable_dict,
                    "overall": {"statistic": overall_stat, "p_value": overall_p},
                }
            return overall_stat, overall_p
        else:
            # For 1D data (used in individual tests)
            try:
                statistic, p_value = ks_2samp(true_data, simulation_data.flatten())
            except Exception:
                statistic, p_value = float("nan"), float("nan")

            print(f"Estadístico KS: {statistic:.4f}")
            print(f"Valor p de KS: {p_value:.4f}")

            if as_dict:
                return {"statistic": float(statistic), "p_value": float(p_value)}
            return float(statistic), float(p_value)

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

        # Perform the Anderson-Darling k-sample test.
        # Recent scipy versions may provide a PermutationMethod helper; call it only if present.
        try:
            perm_cls = getattr(stats, "PermutationMethod", None)
            if perm_cls is not None:
                anderson_result = stats.anderson_ksamp([real_sample, simulated_sample], method=perm_cls())
            else:
                anderson_result = stats.anderson_ksamp([real_sample, simulated_sample])
        except Exception:
            # If anderson_ksamp fails for any reason, return NaNs gracefully
            statistic = float("nan")
            significance_level = float("nan")
        else:
            statistic = float(anderson_result.statistic)
            significance_level = float(getattr(anderson_result, "significance_level", float("nan")))

        print(f"Anderson-Darling Statistic: {statistic:.4f}")
        print(f"Approximate Critical p-value: {significance_level:.3f}")

        if as_dict:
            return {"statistic": statistic, "significance_level": significance_level}
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

        # Use the inverse CDF (ppf) to compute the z-score for the two-sided
        # confidence interval. This avoids calling `norm.interval`, which has
        # changed its parameter names across SciPy versions and can raise
        # unexpected errors. The z for two-sided CI is ppf(1 - alpha/2).
        alpha = 1.0 - coef
        z = norm.ppf(1.0 - alpha / 2.0)

        arr_limite_inferior = mean - z * sem
        arr_limite_superior = mean + z * sem

        # print("conf_int: ", conf_int)
        # print("arr_l_inferior: ", arr_limite_inferior)
        # print("arr_l_superior: ", arr_limite_superior)

        return arr_limite_inferior, arr_limite_superior
