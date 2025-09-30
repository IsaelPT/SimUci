"""Plot histograms comparing true data vs simulated distributions per variable.
Saves PNG files to scripts/out_plots and a CSV summary.
"""

from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from utils.helpers import get_true_data_for_validation, simulate_all_true_data

OUT_DIR = Path("scripts/out_plots")
OUT_DIR.mkdir(parents=True, exist_ok=True)


def main(n_patients=20, n_runs=200, seed=0):
    df = get_true_data_for_validation(seed=seed)
    df_small = df.head(n_patients)
    print(f"Simulating {n_patients} patients x {n_runs} runs (seed={seed})")
    sim_debug = simulate_all_true_data(true_data=df_small, n_runs=n_runs, debug=True, seed=seed)
    if isinstance(sim_debug, dict):
        sim = sim_debug["array"]
    else:
        sim = sim_debug

    # Coerce true data to array matching shape
    td = np.asarray(get_true_data_for_validation(seed=seed))
    if td.ndim == 1 and td.size == sim.shape[0] * sim.shape[2]:
        td = td.reshape((sim.shape[0], sim.shape[2]))
    elif td.ndim == 1 and td.size == sim.shape[2]:
        td = np.tile(td.reshape((1, sim.shape[2])), (sim.shape[0], 1))

    var_names = ["Tiempo Pre VAM", "Tiempo VAM", "Tiempo Post VAM", "Estadia UCI", "Estadia Post UCI"]

    summaries = []

    for i, var in enumerate(var_names):
        sim_values = sim[:, :, i].flatten()
        true_values = td[:, i].flatten()

        fig, ax = plt.subplots(figsize=(6, 4))
        ax.hist(sim_values, bins=30, alpha=0.6, label="Simulated", density=True)
        ax.hist(true_values, bins=30, alpha=0.6, label="True", density=True)
        ax.set_title(var)
        ax.legend()
        out_path = OUT_DIR / f"{i}_{var.replace(' ', '_')}.png"
        fig.savefig(out_path, bbox_inches="tight")
        plt.close(fig)

        summaries.append(
            {
                "variable": var,
                "true_mean": float(np.mean(true_values)),
                "sim_mean": float(np.mean(sim_values)),
                "sim_std": float(np.std(sim_values, ddof=1)),
                "bias": float(np.mean(sim_values) - np.mean(true_values)),
                "true_zero_prop": float((true_values == 0).mean()),
            }
        )

    df_sum = pd.DataFrame(summaries)
    csv_path = OUT_DIR / "summary.csv"
    df_sum.to_csv(csv_path, index=False)
    print(f"Saved plots to {OUT_DIR.resolve()} and summary to {csv_path.resolve()}")


if __name__ == "__main__":
    main()
