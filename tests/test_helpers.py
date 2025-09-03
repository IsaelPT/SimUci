import pandas as pd
import numpy as np
from utils.helpers import build_df_for_stats
from utils.constants import EXPERIMENT_VARIABLES


def make_sim_df(values, n=5):
    # Construye DataFrame con n filas copiando `values`
    data = {col: [val for _ in range(n)] for col, val in values.items()}
    return pd.DataFrame(data)


def test_coverage_count():
    # Crear una simulación donde todas las iteraciones son 10 para cada variable
    vals = {col: 10 for col in EXPERIMENT_VARIABLES}
    df = make_sim_df(vals, n=100)

    # Llamar a build_df_for_stats solicitando intervalos y métricas
    df_out = build_df_for_stats(
        df,
        sample_size=100,
        include_mean=True,
        include_std=True,
        include_confint=True,
        include_metrics=True,
        include_info_label=False,
    )

    # La fila de métricas es la última
    metric_row = df_out.iloc[-1]

    # Dado que todas las simulaciones son constantes, std=0, IC == media == 10, por tanto cobertura = 100 iteraciones
    for col in EXPERIMENT_VARIABLES:
        assert int(metric_row[col]) == 100


def test_coverage_percentage():
    vals = {col: 10 for col in EXPERIMENT_VARIABLES}
    df = make_sim_df(vals, n=50)

    df_out = build_df_for_stats(
        df,
        sample_size=50,
        include_mean=True,
        include_std=True,
        include_confint=True,
        include_metrics=True,
        include_info_label=False,
        metrics_as_percentage=True,
    )
    metric_row = df_out.iloc[-1]

    for col in EXPERIMENT_VARIABLES:
        # 50 de 50 -> 100%
        assert np.isclose(float(metric_row[col]), 100.0)


def test_metrics_reference_scalar():
    # Simulaciones variadas
    vals = {col: 10 for col in EXPERIMENT_VARIABLES}
    df = make_sim_df(vals, n=20)

    # referencia externa con valor 10 para cada columna -> debe indicar 1 (o 100% si se pide porcentaje)
    ref = pd.Series({col: 10 for col in EXPERIMENT_VARIABLES})

    df_out = build_df_for_stats(
        df,
        sample_size=20,
        include_mean=True,
        include_std=True,
        include_confint=True,
        include_metrics=True,
        include_info_label=False,
        metrics_reference=ref,
    )
    metric_row = df_out.iloc[-1]

    for col in EXPERIMENT_VARIABLES:
        assert int(metric_row[col]) == 1
