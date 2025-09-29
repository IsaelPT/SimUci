"""Quick validation script for the simulation model.
Runs a moderate simulation (configurable) and prints coverage/error/KS/AD
and simple per-variable diagnostics (means, bias, zero proportion).

Run with the project's venv python.
"""
import time
import json
import numpy as np
from utils.helpers import get_true_data_for_validation, simulate_all_true_data
from uci.stats import SimulationMetrics

def main(n_patients=12, n_runs=80, seed=None):
    t0 = time.time()
    df = get_true_data_for_validation(seed=seed)
    df_small = df.head(n_patients)
    print(f"Using {df_small.shape[0]} patients, {n_runs} runs each")
    sim = simulate_all_true_data(true_data=df_small, n_runs=n_runs, seed=seed)
    print("Simulated array shape:", sim.shape)
    # Use the same subset of true data that we simulated to avoid mismatches
    true = df_small.reset_index(drop=True)
    sm = SimulationMetrics(true_data=true, simulation_data=sim)
    sm.evaluate()

    print('\nCOVERAGE:')
    print(json.dumps(sm.coverage_percentage, indent=2))
    print('\nERROR_MARGIN:', sm.error_margin)
    print('\nKS:', sm.kolmogorov_smirnov_result)
    print('\nAD:', sm.anderson_darling_result)

    per_patient_means = sim.mean(axis=1)
    sim_means = per_patient_means.mean(axis=0)
    sim_stds = per_patient_means.std(axis=0, ddof=1)

    td = np.asarray(true)
    # Try to coerce shapes similar to internal logic
    if td.ndim == 1 and td.size == sim.shape[0] * sim.shape[2]:
        td = td.reshape((sim.shape[0], sim.shape[2]))
    elif td.ndim == 1 and td.size == sim.shape[2]:
        td = np.tile(td.reshape((1, sim.shape[2])), (sim.shape[0], 1))

    true_means = td.mean(axis=0)

    print('\nTRUE_MEANS:', np.round(true_means, 2))
    print('SIM_MEANS:', np.round(sim_means, 2))
    print('SIM_MEAN_STD_OVER_PATIENTS:', np.round(sim_stds, 2))
    print('BIAS (SIM-TRUE):', np.round(sim_means - true_means, 2))
    print('TRUE_ZERO_PROP:', np.round((td == 0).mean(axis=0), 3))
    print('Elapsed sec:', round(time.time() - t0, 2))

if __name__ == '__main__':
    main()
