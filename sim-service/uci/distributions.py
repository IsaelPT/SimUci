import numpy as np
import pandas as pd
import scipy.stats as stats

from st_utils.constants import RUTA_DFCENTROIDES_CSV


def tiemp_VAM0():
    """Distribuciones para las variables del cluster 0

    Returns:
        val: Muestra aleatoria de la distribucion exponencial
    """

    # Distribuciones para las variables del cluster 0
    lambda_dist = 1 / 113.508
    value = stats.expon.rvs(
        scale=1 / lambda_dist,
    )
    return value


def tiemp_postUCI0():
    """
    Simula una duración de tiempo aleatoria basada en una distribución de probabilidad
    discreta personalizada y distribuciones uniformes para rangos específicos.
    La función utiliza una distribución discreta personalizada con probabilidades
    predefinidas para seleccionar uno de tres posibles resultados (1, 2 o 3).
    Según el resultado seleccionado, genera un valor aleatorio de una distribución
    uniforme dentro de un rango específico:
    - Si el resultado es 1, el valor se genera a partir de una distribución uniforme
      entre 0 y 168.
    - Si el resultado es 2, el valor se genera a partir de una distribución uniforme
      entre 192 y 384.
    - Si el resultado es 3, el valor se genera a partir de una distribución uniforme
      entre 408 y 648.

    Returns:
        numpy.ndarray: Un arreglo de un solo elemento que contiene el valor generado
        aleatoriamente basado en la distribución seleccionada.
    """

    xk = np.array([1, 2, 3])
    pk = np.array([0.6, 0.3, 0.1])

    custom_dist = stats.rv_discrete(name="custom", values=(xk, pk))
    random_numbers = custom_dist.rvs(
        size=1,
    )

    if 1 in random_numbers:
        value = stats.uniform.rvs(
            loc=0,
            scale=168,
            size=1,
        )
        return value
    elif 2 in random_numbers:
        value = stats.uniform.rvs(
            loc=192,
            scale=384,
            size=1,
        )
        return value
    elif 3 in random_numbers:
        value = stats.uniform.rvs(
            loc=408,
            scale=648,
            size=1,
        )
        return value


def estad_UTI0():
    """
    Genera un valor aleatorio basado en una distribución Weibull.
    La función utiliza una distribución Weibull con parámetros específicos
    (forma y escala) para generar un valor aleatorio.
    Returns:
        float: Un valor aleatorio generado a partir de la distribución Weibull.
    """

    forma = 1.37958
    escala = 262.212
    weibull_dist = stats.weibull_min(forma, scale=escala)
    value = weibull_dist.rvs(
        size=1,
    )
    return value


# Distribuciones para las variables del cluster 1
def tiemp_VAM1():
    """
    Genera un valor aleatorio para el tiempo de ventilación mecánica para el cluster 1
    usando una distribución exponencial.

    Returns:
        float: Un valor aleatorio generado a partir de la distribución exponencial
        con lambda = 1/200.
    """

    lambda_dist = 1 / 200
    value = stats.expon.rvs(
        scale=1 / lambda_dist,
    )
    return value


def tiemp_postUCI1():
    """
    Genera un valor aleatorio para el tiempo post-UCI para el cluster 1
    usando una distribución Weibull.

    Returns:
        float: Un valor aleatorio generado a partir de la distribución Weibull
        con forma = 3.63023 y escala = 1214.29.
    """

    forma = 3.63023
    escala = 1214.29
    weibull_dist = stats.weibull_min(forma, scale=escala)
    value = weibull_dist.rvs(
        size=1,
    )
    return value


def estad_UTI1():
    """
    Genera un valor aleatorio para la estadía en UTI para el cluster 1
    usando una distribución Weibull.

    Returns:
        float: Un valor aleatorio generado a partir de la distribución Weibull
        con forma = 1.57768 y escala = 472.866.
    """
    forma = 1.57768
    escala = 472.866
    weibull_dist = stats.weibull_min(forma, scale=escala)
    value = weibull_dist.rvs(
        size=1,
    )
    return value


# Seleccion de cluster
def clustering(
    Edad,
    Diag_Ing1,
    Diag_Ing2,
    Diag_Ing3,
    Diag_Ing4,
    APACHE,
    InsufResp,
    va,
    EstadiaUTI,
    TiempoVAM,
    Est_PreUCI,
):
    """
    Determina el cluster al que pertenece una instancia basándose en sus características
    mediante el cálculo de la distancia euclidiana a los centroides predefinidos.

    Args:
        Edad (float): Edad del paciente
        Diag_Ing1 (int): Diagnóstico de ingreso 1
        Diag_Ing2 (int): Diagnóstico de ingreso 2
        Diag_Ing3 (int): Diagnóstico de ingreso 3
        Diag_Ing4 (int): Diagnóstico de ingreso 4
        APACHE (float): Puntuación APACHE del paciente
        InsufResp (int): Indicador de insuficiencia respiratoria
        va (int): Valor de asistencia ventilatoria
        EstadiaUTI (float): Tiempo de estadía en UTI
        TiempoVAM (float): Tiempo de ventilación mecánica
        Est_PreUCI (float): Estadía pre-UCI

    Returns:
        int: El número del cluster predicho (0 o 1) al que pertenece la instancia
    """
    va_g = 1
    if va == 2 or va == 3:
        va_g = 2

    df_centroid = pd.read_csv(RUTA_DFCENTROIDES_CSV)
    nueva_instancia = np.array(
        [
            Edad,
            Diag_Ing1,
            Diag_Ing2,
            Diag_Ing3,
            Diag_Ing4,
            APACHE,
            InsufResp,
            va,
            va_g,
            EstadiaUTI,
            TiempoVAM,
            Est_PreUCI,
        ]
    )
    distancias = np.linalg.norm(df_centroid.iloc[:, 0:12] - nueva_instancia)
    cluster_predicho = np.argmin(distancias)

    return cluster_predicho
