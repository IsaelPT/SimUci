import secrets
import random
from typing import List, Tuple

import numpy as np
import pandas as pd
from pandas import DataFrame
from streamlit.runtime.uploaded_file_manager import UploadedFile

from utils.constants import (
    CORRIDAS_SIM_DEFAULT,
    VARIABLES_EXPERIMENTO,
    RUTA_MODELO_PREDICCION,
    TIPO_VENT,
    DIAG_PREUCI,
    INSUF_RESP,
)
from uci.experiment import Experiment, multiple_replication
from uci.stats import StatsUtils

import joblib

import sys
import traceback
import streamlit as st


def key_categ(categoria: str, valor: str | int, viceversa: bool = False) -> int | str:
    """
    Obtiene la llave (key, k) que constituye un valor si está presente en la colección de categorías definidas en constants.py.

    Args:
        categoria: Las categorías deben ser entre "va", "diag" e "insuf".
        valor: Es el valor que se pasa para buscar su llave.
        viceversa: Determina si en lugar de buscar el valor, se busca la llave (key).

    Returns:
        Llave que representa en las colecciones de categorías el valor que se pasa por parámetros.
    """

    categorias: dict[int:str]

    if categoria == "va":
        categorias = TIPO_VENT
    elif categoria == "diag":
        categorias = DIAG_PREUCI
    elif categoria == "insuf":
        categorias = INSUF_RESP
    else:
        raise Exception(f"La categoría que se selecciona no existe {categoria}.")

    for k, v in categorias.items():
        if not viceversa:
            if v == valor:
                return k
        else:
            if k == valor:
                return v

    if not viceversa:
        raise Exception(f"El valor (value) que se proporcionó no se encuentra en el conjunto de categorías {categoria}")
    else:
        raise Exception(f"La llave (key) que se proporcionó no se encuentra en el conjunto de categórias {categoria}")


def value_is_zero(valores: list[int | str] | int | str) -> bool:
    """
    Verifica si todos los valores son 0 o "Vacío".

    Args:
        valores: Valor o lista de valores a verificar.

    Returns:
        `true` si el valor o valores son 0 o "vacío", `false` caso contrario.
    """

    def __iszero(v: int | str) -> bool:
        if isinstance(v, int):
            return v == 0
        elif isinstance(v, str):
            return v.lower() == "vacío"

    if isinstance(valores, int | str):
        return __iszero(valores)
    elif isinstance(valores, list):
        return all(__iszero(v) for v in valores)


def generate_id(digits: int = 10) -> str:
    """
    Genera un número pseudoaleatorio de n dígitos. Utilizado para identificar pacientes.

    Args:
        digits: Cantidad de dígitos mayor y diferente de 0 que tendrá el ID. Default = 10 dígitos.

    Returns:
        Cadena de n números generados aleatoriamente.
    """

    if not 0 < digits <= 10:
        raise Exception(f"La cantidad de dígitos n={digits} debe estar en el rango de 0 < d <= 10.")
    return "".join([str(secrets.randbelow(digits)) for _ in range(digits)])


def format_df_time(df: DataFrame, rows_to_format: list[int] = None) -> DataFrame:
    """
    Devuelve una copia del DataFrame con los valores formateados para visualización.
    Los valores se mantienen numéricos en el DataFrame para la serialización.

    Args:
        df: DataFrame a transformar.
        rows_to_format: Lista de índices de filas a formatear. Si es None, formatea todas las filas.

    Returns:
        DataFrame con los valores numéricos originales.
    """
    # Hacer una copia del DataFrame para no modificar el original
    result_df = df.copy()

    # Si no se especifican filas, aplicar a todas
    if rows_to_format is None:
        rows_to_format = list(range(len(df)))

    # Aplicar el formato solo a las filas especificadas
    for idx, row in df.iterrows():
        if idx in rows_to_format:
            for col in df.columns:
                try:
                    # Intentar convertir a float
                    value = float(row[col])
                    if not pd.isna(value):
                        # Mantener el valor numérico original
                        # El formato se aplicará solo al mostrar
                        result_df.at[idx, col] = value
                except (ValueError, TypeError):
                    # Si no se puede convertir a float, mantener el valor original
                    pass

    return result_df


def format_time_columns(
    df: DataFrame,
    exclude_rows: list[str] = None,
) -> DataFrame:
    """
    Formatea las columnas que contienen valores de tiempo en horas.
    Excluye filas específicas del formato.

    Args:
        df: DataFrame con los datos a formatear
        exclude_rows: Lista de nombres de filas a excluir del formato (buscados en la columna 'Información')

    Returns:
        DataFrame con las columnas de tiempo formateadas
    """
    if exclude_rows is None:
        exclude_rows = []

    # Hacer una copia para no modificar el original
    result_df = df.copy()

    # Columnas que suelen contener tiempo
    time_columns = [
        "Tiempo Pre VAM",
        "Tiempo VAM",
        "Tiempo Post VAM",
        "Estadia UCI",
        "Estadia Post UCI",
    ]

    # Filtrar solo las columnas que existen en el DataFrame
    time_columns = [col for col in time_columns if col in result_df.columns]

    # Si no hay columnas de tiempo, retornar el DataFrame sin cambios
    if not time_columns:
        return result_df

    # Verificar si es un DataFrame con columna 'Información'
    has_info_column = "Información" in result_df.columns

    # Aplicar formato a las columnas de tiempo
    for col in time_columns:
        # Crear una copia de la columna y cambiar su tipo a 'object' para evitar FutureWarning
        formatted_col = result_df[col].astype(object)

        # Si hay columna 'Información', buscar las filas a excluir
        if has_info_column:
            for idx in result_df.index:
                # Verificar si la fila actual está en la lista de exclusión
                if result_df.at[idx, "Información"] not in exclude_rows:
                    try:
                        value = result_df.at[idx, col]
                        if pd.notna(value):
                            try:
                                # Convertir a string para evitar problemas de tipo
                                formatted_col.at[idx] = format_value_for_display(value)
                            except (ValueError, TypeError) as e:
                                print(
                                    f"Error al formatear valor {value} en columna '{col}', fila {idx} (Información: {result_df.at[idx, 'Información']}): {str(e)}"
                                )
                    except Exception as e:
                        print(f"Error inesperado al acceder a los datos en columna '{col}', fila {idx}: {str(e)}")
        # Si no hay columna 'Información' pero el índice tiene nombre 'Información'
        elif result_df.index.name == "Información":
            for idx in result_df.index:
                if idx not in exclude_rows:
                    try:
                        value = result_df.at[idx, col]
                        if pd.notna(value):
                            try:
                                # Convertir a string para evitar problemas de tipo
                                formatted_col.at[idx] = format_value_for_display(value)
                            except (ValueError, TypeError) as e:
                                print(f"Error al formatear valor {value} en columna '{col}', índice '{idx}': {str(e)}")
                    except Exception as e:
                        print(f"Error inesperado al acceder a los datos en columna '{col}', índice '{idx}': {str(e)}")
        else:
            # Para DataFrames normales, formatear todas las filas
            formatted_col = result_df[col].apply(lambda x: format_value_for_display(x) if pd.notna(x) else x)

        # Asignar la columna formateada de vuelta, convirtiendo todo a string para evitar ArrowTypeError
        result_df[col] = formatted_col.astype(str)

    return result_df


def format_value_for_display(value: float | int) -> str:
    """
    Formatea un valor numérico como una cadena legible que muestra días y horas.

    Args:
        value: Valor numérico en horas

    Returns:
        Cadena formateada (ej: "1.0 d (24.0 h)" o "0.5 d (30 min)")
    """

    try:
        value = float(value)
        if pd.isna(value):
            return str(value)

        if value >= 1.0:
            return f"{value / 24:.1f} d ({value:.1f} h)"
        else:
            return f"{value / 24:.1f} d ({value * 60:.0f} min)"
    except (ValueError, TypeError):
        return str(value)


def format_df_stats(
    df: DataFrame,
    column_label: str = "Información",
    labels_structure: dict[int | str] | list[str] | None = None,
) -> DataFrame:
    """
    Agrega una columna al extremo izquierdo del DataFrame por parámetros con el nombre de columna "Información".
    Si se proporciona `labels_structure` (dict o list), asigna los labels a las filas correspondientes.

    Comportamiento:
    - Si `labels_structure` es un dict: las claves son índices de fila y los valores son los labels.
    - Si `labels_structure` es una lista: se asigna en orden a las filas 0..len(list)-1.
    - Si `labels_structure` es None o no cubre todas las filas:
        - Para DataFrames con índice tipo RangeIndex y más de una fila (caso de múltiples pacientes), las filas
          sin label se rellenan con "Paciente {i}".
        - En otros casos, las filas sin label se rellenan con "---".

    Args:
        df: DataFrame a aplicar dicho formato.
        column_label: Nombre de la columna a agregar. Default = "Información".
        labels_structure: Estructura de labels (dict índice->label o lista de labels) o None.

    Returns:
        DataFrame con una nueva columna a la izquierda (`column_label`) con labels aplicados.
    """

    # Insertar columna al principio
    df.insert(0, column_label, "")

    unassigned_label = "---"

    n_rows = df.shape[0]

    # Normalizar labels_structure a dict
    labels_map: dict[int, str] = {}
    if labels_structure is None:
        labels_map = {}
    elif isinstance(labels_structure, dict):
        # convertir claves a int si es posible
        for k, v in labels_structure.items():
            try:
                idx = int(k)
            except Exception:
                continue
            labels_map[idx] = str(v)
    elif isinstance(labels_structure, list):
        for i, v in enumerate(labels_structure):
            labels_map[i] = str(v)

    # Asignar labels proporcionados
    for idx, label in labels_map.items():
        if 0 <= idx < n_rows:
            df.at[idx, column_label] = label

    # Rellenar filas sin label
    for i in range(n_rows):
        if df.at[i, column_label] == "":
            # Si parece un DataFrame de múltiples pacientes (filas > 1 y el índice es RangeIndex), usar 'Paciente i'
            if n_rows > 1 and isinstance(df.index, pd.RangeIndex):
                df.at[i, column_label] = f"Paciente {i}"
            else:
                df.at[i, column_label] = unassigned_label

    return df


def build_df_for_stats(
    data: DataFrame | list[DataFrame],
    sample_size: int | None = None,
    include_mean=True,
    include_std=False,
    include_confint=False,
    include_metrics=False,
    include_info_label=True,
    labels_structure: dict[int | str] | list[str] | None = None,
    metrics_as_percentage: bool = False,
    metrics_reference: pd.Series | dict | None = None,
) -> DataFrame:
    """
    Resumen rápido: construye un DataFrame con estadísticas (media, std, intervalos y métricas)
    a partir de un DataFrame (resumen vertical) o una lista de DataFrames (una fila por paciente).

    Args:
      - data: DataFrame o list[DataFrame].
      - sample_size: necesario si include_confint=True.
      - include_*: flags para incluir media, std, confint y métricas.
      - include_info_label: añade la columna 'Información' si True.
      - labels_structure: (opcional) dict {index: label} o list de labels; si se pasa, se usa tal cual.

    Notas clave:
      - Para confint se requieren mean y std y sample_size>0.
      - Si no se pasa labels_structure, se generan labels en el orden lógico según los flags.

    Ejemplo rápido:
      build_df_for_stats(df, include_mean=True, include_std=True, include_confint=True, sample_size=100, include_info_label=True)
    """
    # Implementación resumida y robusta.
    column_label = "Información"

    # Validaciones mínimas
    if include_confint and (not include_mean or not include_std):
        raise ValueError("Para calcular intervalos de confianza se requieren include_mean=True e include_std=True.")
    if include_confint and (sample_size is None or sample_size <= 0):
        raise ValueError("`sample_size` debe ser > 0 cuando include_confint=True.")

    # Si se pasa una lista de DataFrames: producir una fila por paciente con medias
    if isinstance(data, list):
        if not data:
            return pd.DataFrame()

        rows = []
        for df_i in data:
            # Tomar medias de las columnas esperadas; si faltan columnas, pandas llenará con NaN
            rows.append(df_i[VARIABLES_EXPERIMENTO].mean())

        df_out = pd.DataFrame(rows).reset_index(drop=True)

        if include_info_label:
            # Si se pasan labels explícitos, se aplican; en otro caso format_df_stats rellenará 'Paciente i'
            df_out = format_df_stats(df_out, column_label=column_label, labels_structure=labels_structure)

        return df_out

    # Si es un único DataFrame: construir filas verticales según flags
    if isinstance(data, DataFrame):
        df_single = data
        rows = []
        auto_labels: list[str] = []

        if include_mean:
            rows.append(df_single[VARIABLES_EXPERIMENTO].mean())
            auto_labels.append("Promedio")

        if include_std:
            rows.append(df_single[VARIABLES_EXPERIMENTO].std())
            auto_labels.append("Desviación Estándar")

        if include_confint:
            mean = df_single[VARIABLES_EXPERIMENTO].mean()
            std = df_single[VARIABLES_EXPERIMENTO].std()
            li, ls = StatsUtils.confidenceinterval(mean.values, std.values, sample_size)
            # convertir arrays a Series con mismos índices
            li_s = pd.Series(li, index=VARIABLES_EXPERIMENTO)
            ls_s = pd.Series(ls, index=VARIABLES_EXPERIMENTO)
            rows.append(li_s)
            auto_labels.append("Límite Inf")
            rows.append(ls_s)
            auto_labels.append("Límite Sup")

        if include_metrics:
            # Métrica de calibración: contar cuántas iteraciones quedaron dentro del intervalo [LI, LS]
            # Si se proporciona `metrics_reference` como Series/dict con valores reales, se evalúa cobertura real->IC
            # Si `metrics_as_percentage=True`, devolvemos porcentaje en lugar de conteo.
            if include_confint:
                try:
                    counts = {}
                    for col in VARIABLES_EXPERIMENTO:
                        lower = li_s[col]
                        upper = ls_s[col]

                        if metrics_reference is not None:
                            # Si metrics_reference tiene un valor escalar para la columna -> 0/1 si entra en el intervalo
                            if isinstance(metrics_reference, pd.Series) or isinstance(metrics_reference, dict):
                                ref_val = metrics_reference.get(col) if isinstance(metrics_reference, dict) else metrics_reference.get(col, None)
                                if ref_val is None:
                                    counts[col] = 0
                                else:
                                    counts[col] = int(lower <= ref_val <= upper)
                            else:
                                # No soportado, usar placeholder
                                counts[col] = 0
                        else:
                            # Contar filas en el DataFrame original que estén dentro del intervalo
                            counts[col] = int(((df_single[col] >= lower) & (df_single[col] <= upper)).sum())

                    metrics_row = pd.Series([counts.get(col, 0) for col in VARIABLES_EXPERIMENTO], index=VARIABLES_EXPERIMENTO)

                    # Si se solicita porcentaje y estamos trabajando con conteos sobre iteraciones
                    if metrics_as_percentage and metrics_reference is None:
                        denom = df_single.shape[0] if df_single.shape[0] > 0 else 1
                        metrics_row = metrics_row.astype(float) / denom * 100.0
                    elif metrics_as_percentage and metrics_reference is not None:
                        # Cuando comparamos con una referencia escalar, porcentaje es 0 o 100
                        metrics_row = metrics_row.astype(float) * 100.0

                except Exception:
                    # En caso de fallo, usar placeholder
                    metrics_row = pd.Series([0] * len(VARIABLES_EXPERIMENTO), index=VARIABLES_EXPERIMENTO)
            else:
                # Si no se calculó intervalo, no tiene sentido contar cobertura — usar placeholder
                metrics_row = pd.Series([0] * len(VARIABLES_EXPERIMENTO), index=VARIABLES_EXPERIMENTO)

            rows.append(metrics_row)
            auto_labels.append("Métrica de Calibración")

        if not rows:
            raise ValueError("Debe incluir al menos un estadístico (mean/std/confint/metrics).")

        df_out = pd.DataFrame(rows).reset_index(drop=True)

        if include_info_label:
            # Si se pasa labels_structure se aplica; si no, usar auto_labels para un único paciente
            labels_to_use = labels_structure if labels_structure is not None else auto_labels
            df_out = format_df_stats(df_out, column_label=column_label, labels_structure=labels_to_use)

        return df_out

    raise TypeError("`data` debe ser un DataFrame o una lista de DataFrames.")


def bin_to_df(files: UploadedFile | list[UploadedFile]) -> DataFrame | list[DataFrame]:
    """
    Convierte un UploadedFile (un archivo cargado por un file_uploader) en un DataFrame.

    Args:
        files: Archivo o archivos binarios cargados por un file_uploader.

    Returns:
        Archivo o archivos de tipo `.csv` como un DataFrame o lista de DataFrames.
    """

    if isinstance(files, UploadedFile):
        return pd.read_csv(files)
    elif isinstance(files, list):
        return [pd.read_csv(f) for f in files]


def _extract_real_data(ruta_archivo_csv: str, index: int, return_type: str = "df") -> DataFrame | tuple[float]:
    data = pd.read_csv(ruta_archivo_csv)

    def build_row(data_index: int):
        # estuci: días -> horas
        # tiempo_vam: horas
        # estpreuci: días -> horas
        # print(f"TYPE: {type(data_index)} --- DATA: {data_index}\n")
        if isinstance(data_index, int):
            output = {
                "edad": int(data["Edad"].iloc[data_index]),
                "d1": int(data["Diag.Ing1"].iloc[data_index]),
                "d2": int(data["Diag.Ing2"].iloc[data_index]),
                "d3": int(data["Diag.Ing3"].iloc[data_index]),
                "d4": int(data["Diag.Ing4"].iloc[data_index]),
                "apache": int(data["APACHE"].iloc[data_index]),
                "insuf": int(data["InsufResp"].iloc[data_index]),
                "va": int(data["VA"].iloc[data_index]),
                "estuci": int(data["Est. UCI"].iloc[data_index] * 24),
                "tiempo_vam": int(data["TiempoVAM"].iloc[data_index]),
                "estpreuci": int(data["Est. PreUCI"].iloc[data_index] * 24),
            }
            # print(output.values())
            return output
        else:
            raise ValueError("El parámetro data_index debe ser un entero positivo.")

    extracted_data = build_row(index)

    if return_type == "df":
        return pd.DataFrame([extracted_data])  # Default Return Type (DataFrame)
    elif return_type == "tuple":
        return tuple(extracted_data.values())  # Alternative Return Type (Tuple)
    else:
        raise ValueError("El parámetro return_type debe ser 'df' o 'tuple'.")


def start_experiment(
    corridas_simulacion: int,
    edad: int,
    d1: int,
    d2: int,
    d3: int,
    d4: int,
    apache: int,
    insuf_resp: int,
    va: int,
    t_vam: int,
    est_uci: int,
    est_preuti: int,
    porciento: int = 10,
) -> pd.DataFrame:
    """
    Toma una serie de datos de un paciente y con ellos comienza la simulación.

    Args:
        corridas_simulacion: Cantidad de iteraciones que tendrá la simulación.
        edad: Edad del paciente.
        d1: Diagnóstico 1 del paciente.
        d2: Diagnóstico 2 del paciente.
        d3: Diagnóstico 3 del paciente.
        d4: Diagnóstico 4 del paciente.
        apache: Valor del APACHE.
        insuf_resp: Tipo de Insuficiencia Respiratoria que presenta el paciente.
        va: Tiempo de Ventilación Artificial que se espera del paciente.
        t_vam: Tipo de Ventilación que presenta el paciente.
        est_uci: Estadía en UTI que se espera del paciente.
        est_preuti: Estadía Pre-UTI que presenta el paciente.
        porciento: "Proporción de tiempo dentro de estancia UCI que se espera antes de entrar en Ventilación." Por defecto, su valor es 10%.

    Returns:
        Un DataFrame con el resultado de la simulación.

        >>> ["Tiempo Pre VAM", "Tiempo VAM", "Tiempo Post VAM", "Estadia UCI", "Estadia Post UCI"]

    """

    e = Experiment(
        edad=edad,
        diagnostico_ingreso1=d1,
        diagnostico_ingreso2=d2,
        diagnostico_ingreso3=d3,
        diagnostico_ingreso4=d4,
        apache=apache,
        insuficiencia_respiratoria=insuf_resp,
        ventilacion_artificial=va,
        estadia_uti=est_uci,
        tiempo_vam=t_vam,
        tiempo_estadia_pre_uti=est_preuti,
        porciento=porciento,
    )
    # Realizar la simulación
    res = multiple_replication(e, corridas_simulacion)

    # Asegurarse de que todas las columnas sean numéricas
    for col in res.columns:
        # Convertir a numérico, forzando los valores no numéricos a NaN
        res[col] = pd.to_numeric(res[col], errors="coerce")

        # Rellenar NaN con 0 y convertir a entero
        res[col] = res[col].fillna(0).astype("int64")

    # Verificar que no haya valores NaN o None
    if res.isnull().values.any():
        st.warning("Advertencia: Se encontraron valores nulos en los resultados de la simulación.")
        res = res.fillna(0)

    # Asegurarse de que el índice sea secuencial
    res = res.reset_index(drop=True)

    return res


def adjust_df_sizes(dataframes: List[DataFrame]) -> Tuple[List[DataFrame], int]:
    """
    Construye un tuple con una lista de DataFrames las cuales tienen su cantidad de filas iguales al dataframe con menos
    filas en la lista de dataframes de parámetros, y un entero con el valor del size del dataframe con menor size.

    Args:
        dataframes: Lista de DataFrames.

    Returns:
        Tuple con Lista con los DataFrames "recortados" o no, y entero con el valor del size del dataframe más pequeño.
        El entero es -1 si no hubo necesidad de "recortar" los dataframes.

    """

    def all_equal(lst) -> bool | None:
        """Devuelve True si todos los valores de la lista son iguales."""
        if not lst:
            return None
        return len(set(lst)) == 1

    df_sizes: list[int] = [df.shape[0] for df in dataframes]

    try:
        if all_equal(df_sizes):
            return dataframes, -1
        min_len = min(df_sizes)
        return [df.head(min_len) for df in dataframes], min_len
    except Exception() as e:
        print(e)


def build_df_test_result(statistic: float, p_value: float) -> DataFrame:
    """
    Construye un DataFrame con un formato específico que es destinado a la visualización de los resultados de los test estadísticos.

    Args:
        statistic: Valor de Statistics. Resultado de Exámen.
        p_value: Valor de P. Resultado de Exámen.

    Returns:
        DataFrame con datos statistics y p_value que se pasan por parámetros.
    """

    S = "Statistic"
    P = "Valor de P"
    data = {S: [statistic], P: [p_value]}
    df = pd.DataFrame(data)
    return df


def simulate_real_data(ruta_fichero_csv: str, df_selection: int) -> tuple[float] | list[tuple[float]]:
    """Realiza la simulación desde datos reales.

    Args:
        ruta_fichero_csv: Ruta del CSV de datos reales.
        df_selection: Índice del paciente seleccionado; usar -1 para procesar todos los pacientes.

    Returns:
        Tuple con los resultados de la simulación para un paciente, o lista de tuples para todos los pacientes.
    """

    def experiment_helper(t: tuple[float]) -> DataFrame:
        """Construye un DataFrame con los resultados de la simulación.

        Args:
            t (tuple[float]): Tuple con los datos de un paciente.

        Returns:
            DataFrame: DataFrame con los resultados de la simulación.
        """

        e = start_experiment(
            corridas_simulacion=CORRIDAS_SIM_DEFAULT,
            edad=int(t[0]),
            d1=t[1],
            d2=t[2],
            d3=t[3],
            d4=t[4],
            apache=t[5],
            insuf_resp=t[6],
            va=t[7],
            est_uci=t[8],
            t_vam=t[9],
            est_preuti=t[10],
            porciento=random.randint(0, 10),
        )

        return e

    if df_selection != -1:
        t: tuple[float] = _extract_real_data(ruta_fichero_csv, index=df_selection, return_type="tuple")

        # Se retorna un tuple[float]
        return experiment_helper(t)
    elif df_selection == -1:
        datalen = pd.read_csv(ruta_fichero_csv).shape[0]

        # Se retorna un list[tuple[float]]
        return [
            experiment_helper(t)
            for t in [_extract_real_data(ruta_fichero_csv, index=i, return_type="tuple") for i in range(datalen)]
        ]
    else:
        raise ValueError("El parámetro df_selection debe ser -1 o un entero positivo.")


def fix_seed(seed: int = None):
    """Fija la semilla de numpy. Esto es útil para las simulaciones que utilizan una semilla aleatoria basada en el sistema. Al fijar la semilla se pueden obtener resultados con comportamientos específicos. Si la semilla es None, restaura el valor de la semilla aleatoria.

    Args:
        seed (int): Valor máximo 2^32 - 1 (numpy)
        seed (None): Si es None, la semilla vuelve a ser aleatoria

    Raises:
        ValueError: Si se excede el valor máximo de semilla (uint32: 2^32 - 1)
        ValueError: Si la semilla es negativa
    """

    try:
        if seed is not None:
            if seed > np.iinfo(np.int32).max:
                raise ValueError("Se excedió el tamaño de semilla permisible (2^32 - 1)")
            if seed < 0:
                raise ValueError("Semilla debe ser un número entero positivo (uint32)")
            np.random.seed(seed)
        else:
            np.random.seed(None)
    except Exception as e:
        print(e)


def predict(df: DataFrame) -> tuple[np.ndarray, np.ndarray]:
    """
    Realiza una predicción utilizando un modelo previamente entrenado y guardado en 'new_workflow.joblib'.

    Args:
        df (DataFrame): DataFrame con los datos de entrada para la predicción.
            El DataFrame de entrada debe contener únicamente las siguientes columnas:
            {'Diag.Ing1', 'Diag.Ing2', 'Diag.Egr2', 'TiempoVAM', 'APACHE', 'Edad'}

    Returns:
        tuple[np.ndarray, np.ndarray]: Una tupla con el array de predicciones y el array de probabilidades de la clase positiva.

    Ejemplo de resultado devuelto:
    >>> preds, preds_proba = predict(df)
    >>> print(preds)
    [1, 0, 1]
    >>> print(preds_proba)
    [0.85, 0.12, 0.97]

    Raises:
        FileNotFoundError: Si el archivo 'new_workflow.joblib' no se encuentra.
    """

    try:
        # 8/26/2025 - version del modelo entrenada: sklearn - 1.6.1
        model = joblib.load(RUTA_MODELO_PREDICCION)

        preds = model.predict(df)
        preds_proba = model.predict_proba(df)
        res = (preds, np.round(preds_proba[:, 1], 2))

        return res
    except Exception:
        tb_text = "".join(traceback.format_exception(*sys.exc_info()))
        st.error("Ocurrió un error durante la predicción, contacte con los desarrolladores")
        st.code(tb_text, language="python")
        # st.stop()


def get_data_for_prediction(data: dict[str:int] | pd.DataFrame) -> pd.DataFrame:
    """
    Genera un DataFrame con los datos necesarios para realizar una predicción.
    Args:
        data (dict[str, int | float]): Diccionario que contiene los valores de las variables requeridas para la predicción.
            Las claves esperadas son: 'Edad', 'Diag.Ing1', 'Diag.Ing2', 'Diag.Egr2', 'TiempoVAM', 'APACHE'.
    Returns:
        pd.DataFrame: DataFrame con una sola fila y las columnas correspondientes a las variables de entrada.
    Raises:
        Exception: Si ocurre un error durante la construcción del DataFrame, se imprime el traceback y se relanza la excepción.
    """

    # NOTE Required: {'Diag.Ing2', 'Diag.Egr2', 'Diag.Ing1', 'TiempoVAM', 'APACHE', 'Edad'}

    try:
        if isinstance(data, pd.DataFrame):
            required_cols = [
                "Edad",
                "Diag.Ing1",
                "Diag.Ing2",
                "Diag.Egr2",
                "TiempoVAM",
                "APACHE",
            ]
            if not all(col in data.columns for col in required_cols):
                raise ValueError(f"El DataFrame debe contener las columnas: {', '.join(f'{required_cols}')}.")
            return data[required_cols]

        elif isinstance(data, dict):
            diag_ing2 = data.get("Diag.Ing2", 1)
            diag_egr2 = data.get("Diag.Egr2", 2)
            diag_ing1 = data.get("Diag.Ing1", 3)
            tiempo_vam = data.get("TiempoVAM", 4)
            edad = data.get("Edad", 6)
            apache = data.get("APACHE", 5)
            return pd.DataFrame(
                {
                    "Edad": [edad],
                    "Diag.Ing1": [diag_ing1],
                    "Diag.Ing2": [diag_ing2],
                    "Diag.Egr2": [diag_egr2],
                    "TiempoVAM": [tiempo_vam],
                    "APACHE": [apache],
                }
            )

    except Exception as e:
        tb_text = "".join(traceback.format_exception(*sys.exc_info()))
        print(f"Error building prediction data: {e}\n{tb_text}")
        raise
