import secrets
from typing import List, Tuple

import numpy as np
import pandas as pd
from pandas import DataFrame
from streamlit.runtime.uploaded_file_manager import UploadedFile

from st_utils.constants import TIPO_VENT, DIAG_PREUCI, INSUF_RESP
from uci.experiment import Experiment, multiple_replication
from uci.stats import StatsUtils


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

    match categoria:
        case "va":
            categorias = TIPO_VENT
        case "diag":
            categorias = DIAG_PREUCI
        case "insuf":
            categorias = INSUF_RESP
        case _:
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
        raise Exception(f"La cantidad de dígitos n={digits} debe estar en el rango de 0 < d <= 10.")
    return ''.join([str(secrets.randbelow(digits)) for _ in range(digits)])


def format_df_time(df: DataFrame) -> DataFrame:
    """
    Toma todas las columnas y convierte los valores numéricos en texto con formato, mostrando los días y las horas.

    Args:
        df: DataFrame a trasformar..

    Returns:
        DataFrame donde cada celda que contenga números tendrá el formato (# d (# h)).

        Ejemplo:

        >>> {2.0 d (48.0 h)} # -> 2 días, 48 horas.

    """

    def fmt(col: pd.Series):
        mascara: pd.Series = pd.to_numeric(col, errors="coerce").notna()
        formatted: pd.Series = col.astype(object)
        valores_numericos = col[mascara].astype(float)
        formatted[mascara] = [f"{d:.1f} d ({h:.1f}) h" for d, h in zip(valores_numericos / 24, valores_numericos)]
        return formatted

    return df.apply(fmt)


def format_df_stats(df: DataFrame) -> DataFrame:
    """
    Agrega una columna al extremo izquierdo del DataFrame por parámetros con el nombre de columna "Información".
    Brinda de soporte visual para comprender los datos de las tablas.

    Args:
        df: DataFrame a aplicar dicho formato.

    Returns:
        DataFrame con una nueva columna a la izquierda ("Información") con datos estadísticos de utilidad.
    """

    LABEL_INF = "Información"
    df.insert(0, LABEL_INF, "")
    df.loc[0, LABEL_INF] = "Promedio"
    df.loc[1, LABEL_INF] = "Desviación Estándar"
    df.loc[2, LABEL_INF] = "Límite Inferior"
    df.loc[3, LABEL_INF] = "Límite Superior"
    return df


def build_df_stats(df: DataFrame, sample_size: int | None = None,
                   include_info_label=False, include_mean=True, include_std=True, include_confint=True) -> DataFrame:
    if not any([include_mean, include_std, include_confint]):
        raise ValueError("Se debe al menos incluir 1 valor estadísticos.")

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
            raise ValueError(f"Para realizar el intervalo de confianza son necesarios: "
                             f"mean y std: Falta {'mean' if not include_mean else 'std'}")
        else:
            if not sample_size or sample_size <= 0:
                raise ValueError(f"Para realizar el intervalo de confianza debe usarse un tamaño de muestra válido."
                                 f"\ Found: {sample_size}")
            else:
                confint = StatsUtils.confidenceinterval(mean, std, sample_size)
                li = pd.DataFrame(confint[0])
                ls = pd.DataFrame(confint[1])
                li.columns = mean.columns.to_list()
                ls.columns = mean.columns.to_list()
                stats_values.extend([li, ls])

    df_final = pd.concat(stats_values, axis=0, ignore_index=True)

    if include_info_label:
        return format_df_stats(df_final)

    return df_final


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


def extract_real_data(path_datos: str, index: int,
                      return_type: str = "df") -> DataFrame | tuple[float]:
    if index:
        data = pd.read_csv(path_datos)

        row = {
            "edad": int(data["Edad"].iloc[index]),
            "d1": int(data["Diag.Ing1"].iloc[index]),
            "d2": int(data["Diag.Ing2"].iloc[index]),
            "d3": int(data["Diag.Ing3"].iloc[index]),
            "d4": int(data["Diag.Ing4"].iloc[index]),
            "apache": int(data["APACHE"].iloc[index]),
            "insuf": int(data["InsufResp"].iloc[index]),
            "va": int(data["VA"].iloc[index]),
            "estuci": int(data["Est. UCI"].iloc[index] * 24),  # días -> horas
            "tiempo_vam": int(data["TiempoVAM"].iloc[index]),  # horas
            "estpreuci": int(data["Est. PreUCI"].iloc[index] * 24),  # días -> horas
        }

        # Return Type
        if return_type == "tuple":
            return tuple(row.values())
        return pd.DataFrame(data=[row])  # Default Return Type


def start_experiment(corridas_simulacion: int, edad: int, d1: int, d2: int, d3: int, d4: int, apache: int,
                     insuf_resp: int, va: int, t_vam: int, est_uci: int, est_preuti: int, porciento: int = 10
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
        porciento: "Proporción de tiempo dentro de estancia UCI que se espera antes de entrar en Ventilación."

    Returns:
        Un DataFrame con el resultado de la simulación.

        >>> ["Tiempo Pre VAM", "Tiempo VAM", "Tiempo Post VAM", "Estadia UCI", "Estadia Post UCI"]

    """

    e = Experiment(edad=edad, diagnostico_ingreso1=d1, diagnostico_ingreso2=d2, diagnostico_ingreso3=d3,
                   diagnostico_ingreso4=d4, apache=apache, insuficiencia_respiratoria=insuf_resp,
                   ventilacion_artificial=va, estadia_uti=est_uci, tiempo_vam=t_vam, tiempo_estadia_pre_uti=est_preuti,
                   porciento=porciento)
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

    df_sizes = [df.shape[0] for df in dataframes]
    if df_sizes != int(np.mean(df_sizes)):
        min_len = min(df_sizes)
        return [df.head(min_len) for df in dataframes], min_len
    return dataframes, -1


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
    data = {
        S: [statistic],
        P: [p_value]
    }
    df = pd.DataFrame(data)
    return df
