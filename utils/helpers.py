import secrets
import random
from typing import List, Tuple

import numpy as np
import pandas as pd
from pandas import DataFrame
from streamlit.runtime.uploaded_file_manager import UploadedFile

import imaplib

from utils.constants import (
    CORRIDAS_SIM_DEFAULT,
    RUTA_MODELO_PREDICCION,
    TIPO_VENT,
    DIAG_PREUCI,
    INSUF_RESP,
)
from uci.experiment import Experiment, multiple_replication
from uci.stats import StatsUtils

from joblib import load


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
        raise Exception(
            f"El valor (value) que se proporcionó no se encuentra en el conjunto de categorías {categoria}"
        )
    else:
        raise Exception(
            f"La llave (key) que se proporcionó no se encuentra en el conjunto de categórias {categoria}"
        )


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
    else:
        raise ValueError(f"El valor a verificar no es correcto: {valores}")


def generate_id(digits: int = 10) -> str:
    """
    Genera un número pseudoaleatorio de n dígitos. Utilizado para identificar pacientes.

    Args:
        digits: Cantidad de dígitos mayor y diferente de 0 que tendrá el ID. Default = 10 dígitos.

    Returns:
        Cadena de n números generados aleatoriamente.
    """

    if not 0 < digits <= 10:
        raise Exception(
            f"La cantidad de dígitos n={digits} debe estar en el rango de 0 < d <= 10."
        )
    return "".join([str(secrets.randbelow(digits)) for _ in range(digits)])


def format_df_time(df: DataFrame) -> DataFrame:
    """
    Toma todas las columnas y convierte los valores numéricos en texto con formato, mostrando los días y las horas.

    Args:
        df: DataFrame a trasformar.

    Returns:
        DataFrame donde cada celda que contenga números tendrá el formato (# d (# h)).

        Ejemplo:

        >>> {2.0 d (48.0 h)} # -> 2 días, 48 horas.

    """

    def fmt(col: pd.Series):
        mascara: pd.Series = pd.to_numeric(col, errors="coerce").notna()
        format: pd.Series = col.astype(object)  # Para filtrar los valores NaN.
        valores_numericos = col[mascara].astype(float)
        format[mascara] = [
            (
                f"{d:.1f} d ({h:.1f}) h"  # Días en Horas.
                if not 0 < h < 1.0
                else f"{d:.1f} d ({h * 60:.0f} min)"  # Días en Minutos.
            )
            for d, h in zip(valores_numericos / 24, valores_numericos)
        ]
        return format

    return df.apply(fmt)


def format_df_stats(
    df: DataFrame,
    label: str = "Información",
) -> DataFrame:
    """
    Agrega una columna al extremo izquierdo del DataFrame por parámetros con el nombre de columna "Información".
    Brinda de soporte visual para comprender los datos de las tablas.
    Columna:
    >>> [
        "Promedio",
        "Desviación Estándar",
        "Límite Inferior",
        "Límite Superior"
    ]

    Args:
        df: DataFrame a aplicar dicho formato.
        label: Nombre de la columna a agregar. Default = "Información".

    Returns:
        DataFrame con una nueva columna a la izquierda ("Información") con datos estadísticos de utilidad.
    """
    df.insert(0, label, "")

    df.loc[0, label] = "Promedio"
    df.loc[1, label] = "Desviación Estándar"
    df.loc[2, label] = "Límite Inferior"
    df.loc[3, label] = "Límite Superior"
    # for i in range(df.shape[0]):
    #     df.loc[i, label] = f"Paciente {i}"

    return df


def build_df_stats(
    data: DataFrame | list[DataFrame],
    sample_size: int | None = None,
    include_mean=True,
    include_std=False,
    include_confint=False,
    include_info_label=True,
    info_label: str = "Información",
) -> DataFrame:
    """Construye un dataframe que contiene los datos producto de los estudios estadísticos realizados al o a los pacientes.

    Args:
        data (DataFrame | list[DataFrame]): DataFrame o lista de DataFrames con los datos a estudiar.
        sample_size (int | None, optional): Tamaño de muestra. También entendido como cantidad de simulaciones que se tomas en consideración; cantidad de iteraciones. Defaults to None.
        include_info_label (bool, optional): Mostrar en la tabla una columna informativa sobre los campos. Defaults to False.
        include_mean (bool, optional): Mostrar el dato de la media en la tabla. Defaults to True.
        include_std (bool, optional): Mostrar el dato de la desviación estándar. Defaults to True.
        include_confint (bool, optional): Mostrar el dato del intervalor de confianza (límite inf, límite sup). Defaults to True.

    Raises:
        ValueError: Cuando se debe incluir al menos 1 valor estadístico a mostrar.
        ValueError: Cuando al realizar el cálculo de intervalo de confianza no se tengan la media y la desviación estándar.
        ValueError: Cuando al realizar el cálculo de intervalo de confianza se provea un tamaño de muestra incorrecto (x > 0)

    Returns:
        DataFrame: DataFrame incluyendo todos los datos recogidos y calculados.
    """
    if not any([include_mean, include_std, include_confint]):
        raise ValueError("Se debe al menos incluir 1 valor estadísticos.")

    single_patient = True if isinstance(data, DataFrame) else False

    def build_helper(df: DataFrame) -> DataFrame:
        stats_values = []

        # Media
        if include_mean:
            mean: DataFrame = df.mean().to_frame().T
            stats_values.append(mean)

        # Desviación Estándar
        if include_std:
            std: DataFrame = df.std().to_frame().T
            stats_values.append(std)

        # Intervalo de Confianza
        if include_confint:
            if not include_mean and include_std:
                raise ValueError(
                    f"Para realizar el intervalo de confianza son necesarios: "
                    f"mean y std: Falta {'mean' if not include_mean else 'std'}"
                )
            else:
                if not sample_size or sample_size <= 0:
                    raise ValueError(
                        f"Para realizar el intervalo de confianza debe usarse un tamaño de muestra válido."
                        f"Found: {sample_size}"
                    )
                else:
                    confint = StatsUtils.confidenceinterval(mean, std, sample_size)
                    li = pd.DataFrame(confint[0])
                    ls = pd.DataFrame(confint[1])
                    li.columns = mean.columns.to_list()
                    ls.columns = mean.columns.to_list()
                    stats_values.extend([li, ls])

        df_final = pd.concat(stats_values, axis=0, ignore_index=True)

        return df_final

    # Nota: cuando se provee una lista, se hace un promedio de los resultados para cada paciente. Por lo tanto, se debe
    # realizar un promedio de las listas de resultados de cada paciente.
    df_output = (
        build_helper(data)
        if single_patient
        else pd.DataFrame(
            [build_helper(df).iloc[0] for df in data],
            index=[f"Paciente {i}" for i in range(len(data))],
        )
    )

    if include_info_label:
        return format_df_stats(
            df=df_output,
            label=info_label,
        )

    return df_output


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


def extract_real_data(
    ruta_archivo_csv: str, index: int, return_type: str = "df"
) -> DataFrame | tuple[float]:
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
            print(output.values())
            return output
        else:
            raise ValueError("El parámetro data_index debe ser un entero positivo.")

    extracted_data = build_row(index)

    if return_type == "df":
        return pd.DataFrame([extracted_data])  # Alternative Return Type (Tuple)
    elif return_type == "tuple":
        return tuple(extracted_data.values())  # Default Return Type (DataFrame)
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
    res = multiple_replication(e, corridas_simulacion)
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

    df_sizes = [df.shape[0] for df in dataframes]

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


def simulate_real_data(
    ruta_fichero_csv: str, df_selection: int
) -> tuple[float] | list[tuple[float]]:
    """A partir de una porción de los datos reales, realiza una simulación para un paciente seleccionado o para todos los pacientes.

    Args:
        ruta_fichero_csv (str): Ruta del archivo de datos reales.
        df_selection (int): Indice del paciente seleccionado. Si es None, realiza una simulación para todos los pacientes.

    Returns:
        tuple[float] | list[tuple[float]]: Tuple con los resultados de la simulación para un paciente o lista de tuples con los resultados de la simulación para todos los pacientes.
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
        t: tuple[float] = extract_real_data(
            ruta_fichero_csv, index=df_selection, return_type="tuple"
        )

        # Se retorna un tuple[float]
        return experiment_helper(t)
    elif df_selection == -1:
        datalen = pd.read_csv(ruta_fichero_csv).shape[0]

        # Se retorna un list[tuple[float]]
        return [
            experiment_helper(t)
            for t in [
                extract_real_data(ruta_fichero_csv, index=i, return_type="tuple")
                for i in range(datalen)
            ]
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
                raise ValueError(
                    "Se excedió el tamaño de semilla permisible (2^32 - 1)"
                )
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
        ValueError: Si el DataFrame de entrada no tiene las columnas esperadas.
    """

    try:
        model = load(RUTA_MODELO_PREDICCION)
    except Exception as e:
        raise RuntimeError(f"Ocurrió un error al cargar el modelo: {e}")

    try:
        preds = model.predict(df)
        preds_proba = model.predict_proba(df)
    except Exception as e:
        raise RuntimeError(f"Ocurrió un error durante la predicción: {e}")

    return preds, np.round(preds_proba[:, 1], 2)