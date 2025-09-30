from typing import Any, Dict

import matplotlib.pyplot as plt
import numpy as np

from utils.constants import EXPERIMENT_VARIABLES_LABELS


def plot_coverage(coverage: Dict[str, float]):
    labels = list(coverage.keys())
    vals = [coverage[k] for k in labels]

    fig, ax = plt.subplots(figsize=(7, 3.5))
    bars = ax.bar(labels, vals, color="tab:blue", alpha=0.9)
    ax.set_ylim(0, 100)
    ax.set_ylabel("Porcentaje de cobertura (%)")
    ax.set_title("Cobertura de Intervalos de Confianza por Variable")
    ax.axhline(80, color="gray", linestyle="--", alpha=0.6, label="Referencia 80%")
    for bar, v in zip(bars, vals):
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            v + 1,
            f"{v:.1f}%",
            ha="center",
            va="bottom",
            fontsize=8,
        )
    plt.xticks(rotation=40, ha="right")
    plt.tight_layout()
    return fig


def plot_error_margin(error_metric: Dict[str, float] | tuple):
    # Accept either dict or tuple (rmse, mae, mape)
    if isinstance(error_metric, dict):
        rmse = error_metric.get("rmse", float("nan"))
        mae = error_metric.get("mae", float("nan"))
        mape = error_metric.get("mape", float("nan"))
    else:
        try:
            rmse, mae, mape = error_metric
        except Exception:
            rmse = mae = mape = float("nan")

    labels = ["RMSE", "MAE", "MAPE (%)"]
    vals = [rmse, mae, mape]

    fig, ax = plt.subplots(figsize=(6, 3))
    bars = ax.bar(labels, vals, color=["#2b8cbe", "#6baed6", "#08519c"])
    ax.set_title("Resumen de Error")
    for bar, v in zip(bars, vals):
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            v + max(0.01, 0.02 * (abs(v) + 1)),
            f"{v:.2f}",
            ha="center",
            va="bottom",
            fontsize=9,
        )
    plt.tight_layout()
    return fig


def plot_ks(ks_result: Dict[str, Any]):
    # Expecting structure: {"per_variable": {"var_i": {"statistic": s, "p_value": p}}, "overall": {...}}
    per_var = ks_result.get("per_variable", None) if isinstance(ks_result, dict) else None

    if per_var is None:
        # Fallback: try to plot overall statistic only
        overall = ks_result.get("overall", {}) if isinstance(ks_result, dict) else {}
        stat = overall.get("statistic", float("nan"))
        p = overall.get("p_value", float("nan"))
        fig, ax = plt.subplots(figsize=(5, 2))
        ax.text(
            0.5,
            0.5,
            f"KS stat: {stat:.3f}\n p-value: {p:.3f}",
            ha="center",
            va="center",
            fontsize=12,
        )
        ax.axis("off")
        return fig

    labels = []
    stats = []
    pvals = []
    for k, v in per_var.items():
        labels.append(k)
        stats.append(v.get("statistic", float("nan")))
        pvals.append(v.get("p_value", float("nan")))

    fig, ax = plt.subplots(figsize=(7, 3))
    bars = ax.bar(
        labels,
        stats,
        color=["tab:green" if (pv >= 0.05) else "tab:red" for pv in pvals],
    )
    ax.set_ylabel("KS statistic")
    ax.set_title("Prueba Kolmogorov-Smirnov (por variable)")
    plt.xticks(rotation=40, ha="right")
    for bar, s, p in zip(bars, stats, pvals):
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            s + 0.005,
            f"s={s:.3f}\np={p:.3f}",
            ha="center",
            va="bottom",
            fontsize=8,
        )
    plt.tight_layout()
    return fig


def plot_anderson_darling(ad_result: Dict[str, float]):
    stat = ad_result.get("statistic", float("nan")) if isinstance(ad_result, dict) else float("nan")
    sig = ad_result.get("significance_level", float("nan")) if isinstance(ad_result, dict) else float("nan")

    fig, ax = plt.subplots(figsize=(6, 2))
    ax.axis("off")
    txt = f"Anderson-Darling statistic: {stat:.3f}\nApprox. p-value: {sig:.3f}\nInterpretación: "
    if not np.isnan(sig):
        if sig < 0.05:
            txt += "Las distribuciones difieren significativamente (p < 0.05)."
        else:
            txt += "No se detecta diferencia significativa (p ≥ 0.05)."
    else:
        txt += "No disponible."

    ax.text(0.01, 0.5, txt, fontsize=11, va="center")
    plt.tight_layout()
    return fig


def plot_distribution_comparison(true_data, simulation_data, var_names=None):
    # true_data: array shape (n_patients, n_vars) or flattened
    # simulation_data: array shape (n_patients, n_reps, n_vars)
    sim = np.asarray(simulation_data)
    td = np.asarray(true_data)

    if sim.ndim != 3:
        raise ValueError("simulation_data must be 3D array")

    n_vars = sim.shape[2]
    cols = min(3, n_vars)
    rows = int(np.ceil(n_vars / cols))
    fig, axes = plt.subplots(rows, cols, figsize=(cols * 4, rows * 3))
    # Flatten axes to 1D for consistent indexing (handles single-axis cases)
    axes = np.array(axes).ravel()

    for i in range(n_vars):
        ax = axes[i]
        sim_flat = sim[:, :, i].ravel()
        # build reasonable bins based on combined range
        if td.ndim == 2 and td.shape[1] > i:
            true_flat = td[:, i].ravel()
        else:
            true_flat = td.ravel()

        data_comb = np.concatenate([sim_flat, true_flat])
        if data_comb.size == 0:
            ax.text(0.5, 0.5, "No data", ha="center", va="center")
            continue

        bins = max(10, int(min(50, np.ptp(data_comb) // 1 + 1))) if np.ptp(data_comb) > 0 else 10
        ax.hist(
            sim_flat,
            bins=bins,
            alpha=0.6,
            label="Simulado",
            color="tab:orange",
            density=True,
        )
        ax.hist(
            true_flat,
            bins=bins,
            alpha=0.5,
            label="Real",
            color="tab:blue",
            density=True,
        )
        # Prefer provided var_names, otherwise use project labels, otherwise fallback to var_{i}
        if var_names and i < len(var_names):
            name = var_names[i]
        elif i < len(EXPERIMENT_VARIABLES_LABELS):
            name = EXPERIMENT_VARIABLES_LABELS[i]
        else:
            name = f"var_{i}"
        ax.set_title(name)
        ax.legend()

    # turn off extra axes (if any)
    for j in range(n_vars, len(axes)):
        try:
            axes[j].axis("off")
        except Exception:
            pass

    plt.tight_layout()
    return fig


def make_all_plots(sim_metrics, true_data, simulation_data):
    plots = {}
    try:
        plots["coverage"] = plot_coverage(sim_metrics.coverage_percentage or {})
    except Exception:
        plots["coverage"] = None

    try:
        plots["error"] = plot_error_margin(sim_metrics.error_margin or {})
    except Exception:
        plots["error"] = None

    try:
        plots["ks"] = plot_ks(sim_metrics.kolmogorov_smirnov_result or {})
    except Exception:
        plots["ks"] = None

    try:
        plots["ad"] = plot_anderson_darling(sim_metrics.anderson_darling_result or {})
    except Exception:
        plots["ad"] = None

    try:
        plots["distributions"] = plot_distribution_comparison(
            true_data, simulation_data, getattr(sim_metrics, "variable_names", None)
        )
    except Exception:
        plots["distributions"] = None

    return plots


def fig_to_bytes(fig) -> bytes:
    """Convert a Matplotlib figure to PNG bytes for download or display."""
    import io

    buf = io.BytesIO()
    fig.savefig(buf, format="png", dpi=150, bbox_inches="tight")
    buf.seek(0)
    data = buf.getvalue()
    buf.close()
    try:
        plt.close(fig)
    except Exception:
        pass
    return data


def plotly_distribution_chart(true_data, simulation_data, var_index: int, var_name: str | None = None):
    """Return a Plotly Figure comparing the distribution of a single variable.

    - true_data: array-like (n_patients, n_vars) or flattened
    - simulation_data: array-like (n_patients, n_runs, n_vars)
    - var_index: integer index of variable
    - var_name: optional display name
    """
    sim = np.asarray(simulation_data)
    td = np.asarray(true_data)

    if sim.ndim != 3:
        raise ValueError("simulation_data must be 3D array")

    # Extract values
    if td.ndim == 2 and td.shape[1] > var_index:
        true_vals = td[:, var_index].ravel()
    else:
        true_vals = td.ravel()

    sim_vals = sim[:, :, var_index].ravel()

    # Build a combined DataFrame for Plotly
    try:
        import pandas as _pd

        df_true = _pd.DataFrame({"value": true_vals, "source": "Real"})
        df_sim = _pd.DataFrame({"value": sim_vals, "source": "Simulado"})
        df = _pd.concat([df_true, df_sim], ignore_index=True)
    except Exception:
        # Fallback to using numpy arrays with go.Histogram
        df = None

    title = var_name if var_name is not None else f"var_{var_index}"

    # Import plotly lazily so the module can be imported even if plotly is not installed.
    try:
        import plotly.graph_objects as go
    except Exception as e:
        raise ImportError(
            "plotly is required for interactive charts (plotly_distribution_chart). Install with: pip install plotly"
        ) from e

    # Use Plotly's histogram traces with density normalization for overlay
    fig = go.Figure()
    if df is not None:
        fig.add_trace(
            go.Histogram(
                x=df[df["source"] == "Simulado"]["value"],
                histnorm="probability density",
                name="Simulado",
                opacity=0.6,
                marker_color="green",
                nbinsx=40,
            )
        )
        fig.add_trace(
            go.Histogram(
                x=df[df["source"] == "Real"]["value"],
                histnorm="probability density",
                name="Real",
                opacity=0.5,
                marker_color="red",
                nbinsx=40,
            )
        )
    else:
        fig.add_trace(
            go.Histogram(
                x=sim_vals,
                histnorm="probability density",
                name="Simulado",
                opacity=0.6,
                marker_color="green",
            )
        )
        fig.add_trace(
            go.Histogram(
                x=true_vals,
                histnorm="probability density",
                name="Real",
                opacity=0.5,
                marker_color="red",
            )
        )

    # Overlay histograms
    fig.update_layout(
        barmode="overlay",
        title=f"Comparación: {title}",
        xaxis_title=title,
        yaxis_title="Densidad",
    )

    # Add mean lines
    try:
        sim_mean = float(np.nanmean(sim_vals))
        true_mean = float(np.nanmean(true_vals))
        fig.add_vline(
            x=sim_mean,
            line_dash="dash",
            line_color="darkorange",
            annotation_text="Media simulación",
            annotation_position="top left",
        )
        fig.add_vline(
            x=true_mean,
            line_dash="dash",
            line_color="darkblue",
            annotation_text="Media datos real",
            annotation_position="top right",
        )
    except Exception:
        pass

    return fig
