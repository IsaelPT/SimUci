from __future__ import annotations

import math as _math
from typing import Any, Dict

import numpy as _np
import pandas as pd
import streamlit as st

from utils.constants import EXPERIMENT_VARIABLES_LABELS as EXP_VARIABLES
from utils.visuals import (
    plotly_distribution_chart,
    plot_ks,
    fig_to_bytes,
    plot_coverage,
    plot_distribution_comparison,
)


def render_validation(
    simulation_metric: Any,
    true_data: Any,
    simulation_data: Any,
    figs: Dict[str, Any] | None = None,
    figs_bytes: Dict[str, bytes] | None = None,
):
    """Render validation results in Streamlit.

    Minimal, robust renderer that shows error metrics and places a short,
    ICU-focused interpretation of RMSE/MAE/MAPE directly under the metrics.
    """

    figs = figs or {}
    figs_bytes = figs_bytes or {}

    st.markdown("### Resultados de la Validación (resumen)")

    with st.expander("¿Qué significan estas pruebas?", expanded=True):
        st.markdown(
            "Estas pruebas ayudan a evaluar si la simulación reproduce tanto la magnitud como la forma de los datos "
            "observados en la UCI. A continuación un resumen práctico de la información que se usa y qué mide cada prueba:\n\n"
            "- **Datos de entrada**:\n"
            "  - `true_data`: valores observados (habitualmente por paciente).\n"
            "  - `simulation_data`: arreglo con forma `(numero_pacientes, numero_corridas, numero_variables)` que contiene las muestras simuladas.\n"
            "  - Para los tests de error se suele promediar `simulation_data` sobre las corridas por paciente, produciendo `media_por_paciente` con forma `(numero_pacientes, numero_variables)`, y compararla con `true_data`.\n\n"
            "- **Tests de error** (RMSE / MAE / MAPE):\n"
            "  - *RMSE* (Root Mean Squared Error): penaliza más los errores grandes; útil para detectar outliers o colas largas.\n"
            "  - *MAE* (Mean Absolute Error): error medio en las mismas unidades (p. ej. horas); menos sensible a outliers.\n"
            "  - *MAPE* (Mean Absolute Percentage Error): error porcentual; se omiten posiciones donde el valor real es 0 para evitar división por cero.\n\n"
            "- **Cobertura** (intervalos):\n"
            "  - Para cada variable se construye un intervalo alrededor de la media por paciente (por ejemplo usando `t-student` y `ddof=1` cuando hay más de 2 corridas).\n"
            "  - La `Cobertura %` indica qué porcentaje de pacientes tiene su valor observado dentro de ese intervalo. Valores esperados razonables suelen estar en **80–95%** dependiendo de la variable y la incertidumbre.\n\n"
            "- **KS** (Kolmogorov-Smirnov):\n"
            "  - Compara la forma de la distribución: se contrasta la distribución empírica de todas las muestras simuladas (aplanando pacientes * corridas) frente a la distribución real (pacientes).\n"
            "  - Un *p-value* bajo indica diferencias significativas en la forma de las distribuciones (no en la media necesariamente).\n\n"
        )
        st.info(
            "**Notas prácticas**: combinar RMSE/MAE para entender desviaciones en magnitud, revisar la Cobertura para medir incertidumbre/intervalos y use KS para detectar diferencias en la forma de las distribuciones. Anderson–Darling está disponible en el código pero actualmente no se muestra en la UI."
        )

    # Error summary
    with st.container(border=True):
        left, right = st.columns(2, border=False)

        with left:
            st.subheader("Resumen de Error")
            err = getattr(simulation_metric, "error_margin", {}) or {}
            if isinstance(err, dict):
                rmse = err.get("rmse")
                mae = err.get("mae")
                mape = err.get("mape")
            else:
                try:
                    rmse, mae, mape = (err[0], err[1], err[2])
                except Exception:
                    rmse = mae = mape = None

            # Display the metrics (outside the exception block so they always render)
            try:
                st.metric(
                    "RMSE",
                    f"{rmse:.2f}" if rmse is not None else "N/A",
                    help="resalta errores grandes; si es mucho mayor que MAE, busque outliers o colas largas.",
                )
            except Exception:
                st.metric("RMSE", "N/A")

            try:
                st.metric(
                    "MAE",
                    f"{mae:.2f}" if mae is not None else "N/A",
                    help="MAE: error promedio en las mismas unidades (horas).",
                )
            except Exception:
                st.metric("MAE", "N/A")

            try:
                mape_str = f"{mape:.1f}%" if (mape is not None and not (mape != mape)) else "N/A"
                st.metric("MAPE", mape_str, help="MAPE: error porcentual; cuidado si hay muchos ceros.")
            except Exception:
                st.metric("MAPE", "N/A")

        with st.expander("Detalle de cálculos (qué representa cada métrica)", expanded=False):
            # Clear, step-by-step explanation + formulas using LaTeX
            st.markdown("Bloque técnico (cálculo): explicación breve y paso a paso.")

            st.markdown(
                """
                - Notación:
                  - `i = paciente (0..n_pacientes-1)`, v = variable (0..n_variables-1), m = número de corridas.
                  - `\(\bar{x}_{i,v}\)` = media de las m muestras simuladas para el paciente i y variable v.
                  - `\(t_{i,v}\)` = valor real observado para el paciente i y variable v.
                """
            )

            st.markdown("**1) Errores agregados — qué se mide y por qué**")
            st.markdown(
                "Calculamos el error comparando la media por paciente de la simulación \(\bar{x}_{i,v}\) con el dato verdadero \(t_{i,v}\)."
            )

            st.markdown("**RMSE (Root Mean Squared Error)** — destaca errores grandes")
            st.latex(r"RMSE = \sqrt{\frac{1}{N} \sum_{i,v} (\bar{x}_{i,v} - t_{i,v})^{2}}")
            st.markdown(
                "Aquí N normalmente es el número total de pares (i,v) usados en la media: N = n_pacientes \times n_variables (o el subconjunto relevante). RMSE crece con errores grandes."
            )

            st.markdown("**MAE (Mean Absolute Error)** — error medio en las mismas unidades")
            st.latex(r"MAE = \frac{1}{N} \sum_{i,v} |\bar{x}_{i,v} - t_{i,v}|")
            st.markdown("MAE informa el error promedio absoluto; menos sensible a outliers que RMSE.")

            st.markdown("**MAPE (Mean Absolute Percentage Error)** — error relativo (porcentual)")
            st.latex(r"MAPE = 100 \times \frac{1}{N} \sum_{i,v} \frac{|\bar{x}_{i,v} - t_{i,v}|}{t_{i,v}}")
            st.markdown(
                "En la práctica se omiten los términos donde \(t_{i,v} = 0\) para evitar división por cero; si todos los t son cero, MAPE no está definido."
            )

            st.markdown("**2) Cobertura (intervalos de confianza por paciente)** — qué significa")
            st.markdown(
                "Para cada paciente i y variable v usamos las m muestras simuladas para estimar un intervalo alrededor de \(\bar{x}_{i,v}\):"
            )
            st.latex(r"\bar{x}_{i,v} \pm t_{\alpha/2, m-1} \cdot \frac{s_{i,v}}{\sqrt{m}}")
            st.markdown(
                "donde \(s_{i,v}\) es la desviación muestral de las m corridas (use ddof=1 si m>=2) y \(t_{\alpha/2,m-1}\) es el cuantil t para el nivel de confianza elegido."
            )
            st.markdown(
                "La 'Cobertura %' para una variable es el porcentaje de pacientes cuyo valor real \(t_{i,v}\) cae dentro de su correspondiente intervalo. Valores esperables: ~80–95% dependiendo de la incertidumbre."
            )

            st.markdown("**3) KS (Kolmogorov–Smirnov)** — comparar forma de distribuciones")
            st.markdown(
                "Construimos la distribución empírica de las muestras simuladas (aplanando pacientes×corridas) y la comparamos con la distribución empírica de los valores reales (pacientes)."
            )
            st.latex(r"D = \sup_x |F_{sim}(x) - F_{true}(x)|")
            st.markdown(
                "La estadística D mide la máxima discrepancia entre las dos funciones de distribución empírica; el p-value indica si la diferencia es significativa. Atención: KS detecta diferencias en la forma (no sólo en la media)."
            )

            st.markdown(
                "Si quieres ver el código exacto usado para los cálculos numéricos, revisa `uci.stats.SimulationMetrics` donde está la implementación."
            )

        with right:
            try:
                if figs.get("error") is not None:
                    st.pyplot(figs["error"])
                elif figs_bytes.get("error") is not None:
                    st.image(figs_bytes.get("error"))
            except Exception:
                pass

    # Coverage and stats
    c1, c2 = st.columns(2, border=True)

    with c1:
        st.subheader("Cobertura (por variable)")
        # Short caption explaining coverage and its purpose
        st.caption(
            "Cobertura: porcentaje de pacientes cuyo valor real queda dentro del intervalo de confianza estimado a partir de las muestras simuladas. "
            "En nuestro contexto mide si la incertidumbre simulada es coherente con los datos observados por variable."
        )
        cov = getattr(simulation_metric, "coverage_percentage", {}) or {}
        for k, v in cov.items():
            try:
                pct = float(v)
            except Exception:
                pct = 0.0
            st.write(f"{k}: {pct:.1f}%")
            st.progress(min(max(int(pct), 0), 100))
        # Render coverage bar chart (0-100%) for all variables
        try:
            fig_cov = plot_coverage(cov)
            st.pyplot(fig_cov)
            try:
                png_cov = fig_to_bytes(fig_cov)
                st.download_button(
                    "Descargar Cobertura (PNG)",
                    data=png_cov,
                    file_name="coverage_by_variable.png",
                    mime="image/png",
                    use_container_width=True,
                )
            except Exception:
                pass
        except Exception:
            # Fallback: if precomputed figure available in figs, show it
            if figs.get("coverage") is not None:
                try:
                    st.pyplot(figs["coverage"])
                except Exception:
                    if figs_bytes.get("coverage") is not None:
                        st.image(figs_bytes.get("coverage"))

    with c2:
        st.subheader("Pruebas estadísticas (resumen)")
        ks = getattr(simulation_metric, "kolmogorov_smirnov_result", {}) or {}

        def _is_finite_number(x):
            return x is not None and isinstance(x, (int, float)) and _math.isfinite(float(x))

        # If the KS result contains per-variable entries, render a per-variable table and plot
        try:
            if isinstance(ks, dict) and "per_variable" in ks and isinstance(ks["per_variable"], dict):
                per_var = ks["per_variable"]
                # Build DataFrame for display
                rows = []
                for var_name, obj in per_var.items():
                    try:
                        stat = float(obj.get("statistic", float("nan")))
                    except Exception:
                        stat = float("nan")
                    try:
                        p = float(obj.get("p_value", float("nan")))
                    except Exception:
                        p = float("nan")
                    rows.append({"Variable": var_name, "KS_stat": stat, "p_value": p})
                ks_df = pd.DataFrame(rows)
                st.markdown("**KS por variable**")
                # Small explanatory caption: what KS is and why we use it here
                st.caption(
                    "El test de Kolmogorov–Smirnov (KS) mide la máxima diferencia entre dos funciones de distribución empírica. "
                    "En nuestro contexto, sirve para comprobar si la forma de la distribución simulada difiere de la observada por variable, complementando las métricas de magnitud."
                )
                st.dataframe(ks_df, use_container_width=True)

                # Render the KS bar plot (green/red by p-value)
                try:
                    fig_ks = plot_ks(ks)
                    st.pyplot(fig_ks)
                    try:
                        png = fig_to_bytes(fig_ks)
                        st.download_button(
                            "Descargar KS (PNG)",
                            data=png,
                            file_name="ks_by_variable.png",
                            mime="image/png",
                            use_container_width=True,
                        )
                    except Exception:
                        pass
                except Exception:
                    st.write("No se pudo generar el gráfico KS (intente instalar dependencias de plotting).")
            else:
                # Fallback: show overall KS p-value if available
                ks_p = None
                if isinstance(ks, dict):
                    overall = ks.get("overall") or {}
                    ks_p = overall.get("p_value") if isinstance(overall, dict) else None
                if _is_finite_number(ks_p):
                    st.write(f"KS global p={float(ks_p):.3f}")
                else:
                    st.write("KS: no disponible")
        except Exception:
            st.write("KS: no disponible")

        # Anderson-Darling: currently disabled (kept commented for future use)
        # try:
        #     if isinstance(ad, dict) and ("statistic" in ad or "significance_level" in ad):
        #         st.markdown("**Anderson-Darling (resumen)**")
        #         try:
        #             fig_ad = plot_anderson_darling(ad)
        #             st.pyplot(fig_ad)
        #             try:
        #                 png_ad = fig_to_bytes(fig_ad)
        #                 st.download_button("Descargar AD (PNG)", data=png_ad, file_name="ad_summary.png", mime="image/png")
        #             except Exception:
        #                 pass
        #         except Exception:
        #             # If plotting fails, show numeric summary
        #             sig = ad.get("significance_level") if isinstance(ad, dict) else None
        #             if _is_finite_number(sig):
        #                 st.write(f"AD p≈{float(sig):.3f}")
        #             else:
        #                 st.write("AD: no disponible")
        #     else:
        #         st.write("AD: no disponible")
        # except Exception:
        #     st.write("AD: no disponible")

    st.markdown("---")
    st.markdown("### Comparación de distribuciones (simulado vs. real)")

    try:
        var_names = list(EXP_VARIABLES)
    except Exception:
        var_names = None

    if simulation_data is not None:
        try:
            n_vars = int(simulation_data.shape[2])
        except Exception:
            n_vars = len(var_names) if var_names is not None else 0

        options = [(i, var_names[i] if var_names and i < len(var_names) else f"var_{i}") for i in range(n_vars)]
        if options:
            labels = [f"{i}: {name}" for i, name in options]
            sel = st.selectbox("Seleccionar variable", labels, index=0, key="select_dist_var")
            idx = int(sel.split(":", 1)[0])
            name = var_names[idx] if var_names and idx < len(var_names) else f"var_{idx}"

            st.caption(f"Variable seleccionada: {idx} — {name}")
            try:
                fig = plotly_distribution_chart(true_data, simulation_data, idx, var_name=name)
                st.plotly_chart(fig, use_container_width=True)
            except Exception:
                if figs.get("distributions") is not None:
                    try:
                        st.pyplot(figs["distributions"])
                    except Exception:
                        if figs_bytes.get("distributions") is not None:
                            st.image(figs_bytes.get("distributions"))
                elif figs_bytes.get("distributions") is not None:
                    st.image(figs_bytes.get("distributions"))

            # Provide a download button for the full Matplotlib figure (all variables) as PNG
            try:
                fig_all = plot_distribution_comparison(true_data, simulation_data, var_names)
                png_all = fig_to_bytes(fig_all)
                st.download_button(
                    "Descargar todas las distribuciones (PNG)",
                    data=png_all,
                    file_name="distributions_all_vars.png",
                    mime="image/png",
                    use_container_width=True,
                )
            except Exception:
                # If generation fails, silently skip the download button
                pass

            # quick debug stats
            try:
                sim_vals = simulation_data[:, :, idx].ravel()
                td_arr = _np.asarray(true_data)
                if td_arr.ndim == 2 and idx < td_arr.shape[1]:
                    true_vals = td_arr[:, idx]
                else:
                    true_vals = td_arr.ravel()

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
    else:
        st.write("No hay datos de simulación para mostrar distribuciones.")

    # Diagnostics table
    try:
        per_patient_means = simulation_data.mean(axis=1)
        sim_means = per_patient_means.mean(axis=0)
        sim_stds = (
            per_patient_means.std(axis=0, ddof=1) if per_patient_means.shape[0] > 1 else _np.zeros_like(sim_means)
        )

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
                "Cobertura %": [
                    getattr(simulation_metric, "coverage_percentage", {}).get(v, None) for v in EXP_VARIABLES
                ],
            }
        )

        st.markdown("### Diagnóstico por variable")
        st.dataframe(diag_df, use_container_width=True)
    except Exception as e:
        st.write(f"Could not build diagnostics table: {e}")
