import secrets

import pandas as pd
from pandas import DataFrame
from streamlit.runtime.uploaded_file_manager import UploadedFile

from constants import TIPO_VENT, DIAG_PREUCI, INSUF_RESP


def key_categ(categoria: str, valor: str | int, viceversa: bool = False) -> int | str:
    """
    Obtiene la llave (key, k) que constituye un valor si está presente en la colección de categorías definidas.

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
    if isinstance(valores, int | str):
        return __iszero(valores)
    elif isinstance(valores, list):
        return all(__iszero(v) for v in valores)
    else:
        raise ValueError(f"El valor a verificar no es correcto: {valores}")


def __iszero(v: int | str) -> bool:
    if isinstance(v, int):
        return v == 0
    elif isinstance(v, str):
        return v.lower() == "vacío"


def generate_id(n: int = 10) -> str:
    """
    Genera un número pseudoaleatorio de n dígitos. Utilizado para identificar pacientes.

    Args:
        n: Cantidad de dígitos mayor y diferente de 0 que tendrá el ID. Default = 10 dígitos.

    Returns:
        Cadena de n números generados aleatoriamente.
    """
    if n != 0:
        return ''.join([str(secrets.randbelow(n)) for _ in range(n)])
    else:
        raise Exception(f"La cantidad de dígitos n={n} debe ser mayor distinta que 0.")


def format_df(datos: DataFrame, enhance_format: bool = False, data_at_beginning: bool = False) -> DataFrame:
    """
    Construye un nuevo DataFrame. Agrega al comienzo del dataframe el *promedio* y *desviación estándar* de todos los valores.

    Args:
        data_at_beginning: Muestra los datos nuevos al principio del dataframe. En caso contrario, los muestra al final.
        datos: DataFrame base.
        enhance_format: Usar solo si es para mostrar datos. Para cada número en la tabla el carácter "h" para expresar que los números están expresados en *horas*.

    Returns:
        DataFrame nuevo con nuevas filas de promedio y desviación estándar con los valores del DataFrame.
    """

    # Construir DataFrame (salida)
    nuevos_datos = {
        "Promedio": datos.mean(),
        "Desviación Estándar": datos.std(),
        "Intervalo Confianza": datos.std(),  # PROVISIONAL
    }
    n_datos_values = list(nuevos_datos.values())
    n_datos_labels = list(nuevos_datos.keys())

    df_nuevos_datos = pd.DataFrame(n_datos_values, index=n_datos_labels)

    len_datos = datos.shape[0]
    len_nuevos_datos = df_nuevos_datos.shape[0]

    # Construir Labels y Columna Informativa.
    LABEL_INF = "Información"

    def build_labels_helper():
        res.insert(0, LABEL_INF, "")
        for index, label in enumerate(n_datos_labels):
            print(index, ":", label)
            if data_at_beginning:
                res.loc[index, LABEL_INF] = label
            else:
                res.loc[len_datos + index, LABEL_INF] = label

    if data_at_beginning:
        res = pd.concat([df_nuevos_datos, datos], axis=0).reset_index(drop=True)
        build_labels_helper()
        for i in range(len_nuevos_datos, len_datos + len_nuevos_datos):
            res.loc[i, LABEL_INF] = f"Iteración {i - len_nuevos_datos + 1}"
    else:
        res = pd.concat([datos, df_nuevos_datos], axis=0).reset_index(drop=True)
        build_labels_helper()
        for i in range(0, len_datos):
            res.loc[i, LABEL_INF] = f"Iteración {i + 1}"

    # Formato
    if enhance_format:
        def fmt(horas: int | float) -> str | int:
            if isinstance(horas, (int, float)):
                return f"{horas / 24:.1f} días ({horas:.1f} h)"
            return horas

        res = res.applymap(fmt)

    return res


def bin_to_df(bin_file: UploadedFile) -> DataFrame:
    """
    Convierte un UploadedFile (un archivo cargado por un file_uploader) en un DataFrame.

    Args:
        bin_file: Archivo binario cargado por un file_uploader.

    Returns:
        Ese archivo (debe ser `.csv`) como un DataFrame.
    """

    return pd.read_csv(bin_file)
