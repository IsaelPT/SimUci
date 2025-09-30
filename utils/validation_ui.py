from __future__ import annotations

import math as _math
from typing import Any, Dict

import numpy as _np
import pandas as pd
import streamlit as st

from utils.constants import EXPERIMENT_VARIABLES_LABELS as EXP_VARIABLES
from utils.visuals import plotly_distribution_chart


def render_validation(
    simulation_metric: Any,
    true_data: Any,
    simulation_data: Any,
    figs: Dict[str, Any] | None = None,
    figs_bytes: Dict[str, bytes] | None = None,
):
    """Render validation results given computed metrics and figures.

    This was extracted from `app.py` so the Streamlit entrypoint file can stay
    mainly declarative. It mirrors the previous behaviour but ensures the
    metric columns are rendered with borders.
    """

    figs = figs or {}
    figs_bytes = figs_bytes or {}

    st.markdown("### Resultados de la Validación (resumen)")

    with st.expander("¿Qué significan estas pruebas?", expanded=True):
        st.markdown(
            (
                "- **Cobertura (Coverage)**: porcentaje de pacientes cuya verdad cae dentro del "
                "intervalo de confianza estimado. Valores cercanos a 80-95% indican que los "
                "intervalos son bien calibrados. La barra indica el porcentaje por variable.\n"
                "- **Resumen de Error (RMSE / MAE / MAPE)**: medidas de ajuste entre medias simuladas "
                "y valores reales. Valores más bajos indican mejor ajuste.\n"
                "  - RMSE (Root Mean Squared Error): raíz del promedio de los errores al cuadrado; "
                "penaliza errores grandes y se expresa en las mismas unidades que la variable (aquí, horas).\n"
                "  - MAE (Mean Absolute Error): promedio de las magnitudes absolutas de los errores; "
                "menos sensible a valores extremos y también se expresa en horas.\n"
                "  - MAPE (Mean Absolute Percentage Error): error absoluto promedio expresado como porcentaje "
                "respecto al valor real; útil para entender la magnitud relativa del error. En esta implementación, "
                "los casos con valor real cero se omiten para evitar división por cero.\n"
                "- **Prueba Kolmogorov–Smirnov (KS)**: compara la forma de la distribución real vs. "
                "simulada por variable. Un p-valor >= 0.05 sugiere que no se detectan diferencias significativas.\n"
                "- **Prueba Anderson–Darling (AD)**: otra prueba de bondad de ajuste; p-valor aproximado "
                "< 0.05 indica diferencias significativas entre distribuciones.\n\n"
                "Colores y símbolos usados:\n"
                "- Verde: no se detecta diferencia estadística (p ≥ 0.05).\n"
                "- Rojo: diferencia estadística detectada (p < 0.05).\n\n"
                "Interprete estas pruebas como señales: p < 0.05 sugiere que la simulación y los datos "
                "reales podrían diferir en la forma de la distribución; no es una prueba de causalidad, "
                "sólo una alerta para investigar más."
            )
        )

    # Top row: three metric cards (render with borders)
    card1, card2, card3 = st.columns(3, border=True)

    # Error metrics card
    with card1:
        st.subheader("Resumen de Error")
        err = getattr(simulation_metric, "error_margin", {}) or {}
        rmse = (
            err.get("rmse")
            if isinstance(err, dict)
            else (err[0] if isinstance(err, (list, tuple)) and len(err) > 0 else None)
        )
        mae = (
            err.get("mae")
            if isinstance(err, dict)
            else (err[1] if isinstance(err, (list, tuple)) and len(err) > 1 else None)
        )
        mape = (
            err.get("mape")
            if isinstance(err, dict)
            else (err[2] if isinstance(err, (list, tuple)) and len(err) > 2 else None)
        )
        st.metric(label="RMSE", value=f"{rmse:.2f}" if rmse is not None else "N/A")
        st.metric(label="MAE", value=f"{mae:.2f}" if mae is not None else "N/A")
        st.metric(
            label="MAPE",
            value=(f"{mape:.1f}%" if (mape is not None and not (mape != mape)) else "N/A"),
        )
        if figs.get("error") is not None or figs_bytes.get("error") is not None:
            try:
                if figs.get("error") is not None:
                    st.pyplot(figs["error"])
                else:
                    st.image(figs_bytes.get("error"))
            except Exception:
                if figs_bytes.get("error") is not None:
                    st.image(figs_bytes.get("error"))
            img_dl = figs_bytes.get("error")
            if img_dl:
                st.download_button(
                    "Descargar gráfico de error",
                    data=img_dl,
                    file_name="error_plot.png",
                    mime="image/png",
                    key="download_error_plot_1",
                )

    # Coverage card
    with card2:
        st.subheader("Cobertura (por variable)")
        cov = getattr(simulation_metric, "coverage_percentage", {}) or {}
        for k, v in cov.items():
            try:
                pct = float(v)
            except Exception:
                pct = 0.0
            st.write(f"{k}: {pct:.1f}%")
            st.progress(min(max(int(pct), 0), 100))
        if figs.get("coverage") is not None or figs_bytes.get("coverage") is not None:
            try:
                if figs.get("coverage") is not None:
                    st.pyplot(figs["coverage"])
                else:
                    st.image(figs_bytes.get("coverage"))
            except Exception:
                if figs_bytes.get("coverage") is not None:
                    st.image(figs_bytes.get("coverage"))
            img_dl = figs_bytes.get("coverage")
            if img_dl:
                st.download_button(
                    "Descargar cobertura",
                    data=img_dl,
                    file_name="coverage_plot.png",
                    mime="image/png",
                    key="download_coverage_plot_1",
                )

    # Statistical tests card
    with card3:
        st.subheader("Pruebas estadísticas")
        ks = getattr(simulation_metric, "kolmogorov_smirnov_result", {}) or {}
        ad = getattr(simulation_metric, "anderson_darling_result", {}) or {}

        def _is_finite_number(x):
            return x is not None and isinstance(x, (int, float)) and _math.isfinite(float(x))

        def badge(ok: bool, text_ok: str, text_bad: str):
            if ok:
                st.markdown(
                    f"<span style='color:green;font-weight:bold'>{text_ok}</span>",
                    unsafe_allow_html=True,
                )
            else:
                st.markdown(
                    f"<span style='color:red;font-weight:bold'>{text_bad}</span>",
                    unsafe_allow_html=True,
                )

        # KS summary
        ks_overall_p = None
        try:
            if isinstance(ks, dict):
                overall = ks.get("overall") or {}
                ks_overall_p = overall.get("p_value") if isinstance(overall, dict) else None
        except Exception:
            ks_overall_p = None

        if _is_finite_number(ks_overall_p):
            ks_overall_p = float(ks_overall_p)
            badge(
                ks_overall_p >= 0.05,
                f"KS: p={ks_overall_p:.3f} (no diferencia detectada)",
                f"KS: p={ks_overall_p:.3f} (diferencia detectada)",
            )
        else:
            per_var = ks.get("per_variable") if isinstance(ks, dict) else None
            if isinstance(per_var, dict) and len(per_var) > 0:
                pvals = []
                for kname, obj in per_var.items():
                    try:
                        pv = float(obj.get("p_value", float("nan")))
                    except Exception:
                        pv = float("nan")
                    pvals.append(pv)
                pvals_arr = _np.array(pvals, dtype=float)
                finite_mask = _np.isfinite(pvals_arr)
                if finite_mask.any():
                    ok_count = int((pvals_arr[finite_mask] >= 0.05).sum())
                    total = int(finite_mask.sum())
                    st.write(f"KS per-variable: {ok_count}/{total} variables sin diferencia detectada (p≥0.05).")
                else:
                    st.write("KS: no disponible")
            else:
                st.write("KS: no disponible")

        # AD summary
        ad_sig = None
        try:
            if isinstance(ad, dict):
                ad_sig = ad.get("significance_level")
        except Exception:
            ad_sig = None

        if _is_finite_number(ad_sig):
            ad_sig = float(ad_sig)
            badge(
                ad_sig >= 0.05,
                f"AD: p≈{ad_sig:.3f} (no diferencia detectada)",
                f"AD: p≈{ad_sig:.3f} (diferencia detectada)",
            )
        else:
            st.write("AD: no disponible")

        # KS/AD plots
        if figs.get("ks") is not None or figs_bytes.get("ks") is not None:
            try:
                if figs.get("ks") is not None:
                    st.pyplot(figs["ks"])
                else:
                    st.image(figs_bytes.get("ks"))
            except Exception:
                if figs_bytes.get("ks") is not None:
                    st.image(figs_bytes.get("ks"))
            img_dl = figs_bytes.get("ks")
            if img_dl:
                st.download_button(
                    "Descargar KS",
                    data=img_dl,
                    file_name="ks_plot.png",
                    mime="image/png",
                    key="download_ks_plot_1",
                )
        if figs.get("ad") is not None or figs_bytes.get("ad") is not None:
            try:
                if figs.get("ad") is not None:
                    st.pyplot(figs["ad"])
                else:
                    st.image(figs_bytes.get("ad"))
            except Exception:
                if figs_bytes.get("ad") is not None:
                    st.image(figs_bytes.get("ad"))
            img_dl = figs_bytes.get("ad")
            if img_dl:
                st.download_button(
                    "Descargar AD",
                    data=img_dl,
                    file_name="ad_plot.png",
                    mime="image/png",
                    key="download_ad_plot_1",
                )

    st.markdown("### Comparación de distribuciones (simulado vs. real)")
    try:
        legend_items = [f"{i}: {name}" for i, name in enumerate(EXP_VARIABLES)]
        st.caption("Variables — índice: nombre — " + ", ".join(legend_items))
    except Exception:
        pass

    # Prefer interactive Plotly per-variable view. Provide a selector to explore each variable.
    try:
        var_names = list(EXP_VARIABLES)
    except Exception:
        var_names = None

    # Distribution comparisons // Small help expander explaining what the distribution comparison shows and how to interpret it
    with st.expander("¿Qué muestra la comparación de distribuciones?", expanded=False):
        st.markdown(
            (
                "Aquí puede comparar la distribución empírica de una variable en los datos reales "
                "frente a la distribución generada por la simulación. Seleccione la variable en el selector "
                "para ver un gráfico interactivo (densidad/histograma) que superpone las dos distribuciones.\n\n"
                "Qué observar:\n"
                "- Si las curvas se solapan mucho, la simulación reproduce bien la forma de la distribución real.\n"
                "- Desviaciones sistemáticas (picos desplazados o colas más largas) indican aspectos del modelo a revisar.\n"
                "- Use las pruebas estadísticas (KS, AD) y las métricas de error (RMSE/MAE/MAPE) para cuantificar diferencias y no sólo basarse en la apariencia visual.\n\n"
                "Consejos rápidos:\n"
                "- Si MAPE es alto pero RMSE es bajo, los errores relativos son grandes en variables pequeñas; revise escala y ceros.\n"
                "- Si KS/AD devuelven p < 0.05 para la variable, investigue sesgos, outliers o diferencias en la cola de la distribución.\n"
                "- Use el panel de diagnóstico por variable (tabla) para ver medias, desviaciones, bias y proporción de ceros."
            )
        )

    if simulation_data is not None:
        try:
            n_vars = int(simulation_data.shape[2])
        except Exception:
            n_vars = len(var_names) if var_names is not None else None

        # Build selector labels
        options = []
        for i in range(n_vars if n_vars is not None else 0):
            label = var_names[i] if (var_names and i < len(var_names)) else f"var_{i}"
            options.append((i, label))

        # Show a selectbox with human-friendly labels
        selected_idx = 0
        if options:
            labels_for_select = [f"{i}: {name}" for i, name in options]
            # Use an explicit key to avoid Streamlit auto-ID collisions across reruns
            sel = st.selectbox(
                "Seleccionar variable",
                labels_for_select,
                index=0,
                key="select_distribution_var",
            )
            selected_idx = int(sel.split(":", 1)[0])

        # Render Plotly interactive chart for the selected variable
        try:
            name = var_names[selected_idx] if (var_names and selected_idx < len(var_names)) else f"var_{selected_idx}"
            st.caption(f"Variable seleccionada: {selected_idx} — {name}")
            plotly_fig = plotly_distribution_chart(true_data, simulation_data, selected_idx, var_name=name)
            st.plotly_chart(plotly_fig, use_container_width=True)
        except Exception:
            # Fallback to existing matplotlib/image if Plotly fails
            if figs.get("distributions") is not None:
                try:
                    st.pyplot(figs["distributions"])
                except Exception:
                    if figs_bytes.get("distributions") is not None:
                        st.image(figs_bytes.get("distributions"))
            elif figs_bytes.get("distributions") is not None:
                st.image(figs_bytes.get("distributions"))
            else:
                st.write("No hay plot de distribuciones disponible")

        # Small diagnostics to help debug if different variables look identical
        try:
            sim_vals = simulation_data[:, :, selected_idx].ravel()
            if hasattr(true_data, "to_numpy"):
                td_arr = _np.asarray(true_data)
                if td_arr.ndim == 2 and selected_idx < td_arr.shape[1]:
                    true_vals = td_arr[:, selected_idx]
                else:
                    true_vals = td_arr.ravel()
            else:
                true_vals = _np.asarray(true_data)[:, selected_idx]

            with st.expander("Debug: estadísticas de la variable seleccionada", expanded=False):
                st.write(
                    {
                        "sim_count": int(sim_vals.size),
                        "sim_min": float(_np.min(sim_vals)),
                        "sim_max": float(_np.max(sim_vals)),
                        "sim_mean": float(_np.round(_np.mean(sim_vals), 3)),
                        "true_unique": list(_np.unique(true_vals)[:10]),
                        "true_count": int(_np.size(true_vals)),
                    }
                )
        except Exception:
            pass

        # Keep PNG download for backwards compatibility (if available)
        img_dl = figs_bytes.get("distributions")
        if img_dl:
            st.download_button(
                "Descargar comparaciones",
                data=img_dl,
                file_name="dist_compare.png",
                mime="image/png",
                key="download_dist_compare_1",
            )
    else:
        # No simulation data available: try to show the precomputed image if present
        if figs.get("distributions") is not None:
            try:
                st.pyplot(figs["distributions"])
            except Exception:
                if figs_bytes.get("distributions") is not None:
                    st.image(figs_bytes.get("distributions"))
        elif figs_bytes.get("distributions") is not None:
            st.image(figs_bytes.get("distributions"))
        else:
            st.write("No hay plot de distribuciones disponible")

    # Per-variable diagnostics table
    try:
        per_patient_means = simulation_data.mean(axis=1)
        sim_means = per_patient_means.mean(axis=0)
        sim_stds = per_patient_means.std(axis=0, ddof=1)

        td = _np.asarray(true_data)
        if td.ndim == 1 and td.size == simulation_data.shape[0] * simulation_data.shape[2]:
            td = td.reshape((simulation_data.shape[0], simulation_data.shape[2]))
        elif td.ndim == 1 and td.size == simulation_data.shape[2]:
            td = _np.tile(td.reshape((1, simulation_data.shape[2])), (simulation_data.shape[0], 1))

        true_means = td.mean(axis=0)
        bias = sim_means - true_means
        zero_prop = (_np.asarray(td) == 0).mean(axis=0)

        diag_df = pd.DataFrame(
            {
                "Variable": EXP_VARIABLES,
                "Media Real": _np.round(true_means, 2),
                "Media Sim": _np.round(sim_means, 2),
                "Desv. Est. Sim (sobre pacientes)": _np.round(sim_stds, 2),
                "Bias (Desviación Sim-Real)": _np.round(bias, 2),
                "Proporción valores Cero": _np.round(zero_prop, 3),
                "Covertura %": [
                    getattr(simulation_metric, "coverage_percentage", {}).get(v, None) for v in EXP_VARIABLES
                ],
            }
        )

        st.markdown("### Diagnóstico por variable")
        st.dataframe(diag_df, use_container_width=True)
    except Exception as e:
        st.write(f"Could not build diagnostics table: {e}")
