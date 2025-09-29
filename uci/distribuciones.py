import numpy as np
import pandas as pd
import scipy.stats as stats

from utils.constants import DFCENTROIDES_CSV_PATH


# Distribuciones para las variables del cluster 0
def tiemp_VAM_cluster0():
    lambda_dist = 1 / 113.508
    value = stats.expon.rvs(scale=1 / lambda_dist)
    return value


def tiemp_postUCI_cluster0():
    xk = np.array([1, 2, 3])
    pk = np.array([0.6, 0.3, 0.1])

    custom_dist = stats.rv_discrete(name="custom", values=(xk, pk))
    random_numbers = custom_dist.rvs(size=1)

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


def estad_UTI_cluster0():
    forma = 1.37958
    escala = 262.212
    weibull_dist = stats.weibull_min(forma, scale=escala)
    value = weibull_dist.rvs(size=1)
    return value


# Distribuciones para las variables del cluster 1
def tiemp_VAM_cluster1():
    lambda_dist = 1 / 200
    value = stats.expon.rvs(scale=1 / lambda_dist)
    return value


def tiemp_postUCI_clustet1():
    forma = 3.63023
    escala = 1214.29
    weibull_dist = stats.weibull_min(forma, scale=escala)
    value = weibull_dist.rvs(size=1)
    return value


def estad_UTI_cluster1():
    forma = 1.57768
    escala = 472.866
    weibull_dist = stats.weibull_min(forma, scale=escala)
    value = weibull_dist.rvs(size=1)
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
) -> np.intp:
    va_g = 1
    if va == 2 or va == 3:
        va_g = 2

    df_centroid = pd.read_csv(DFCENTROIDES_CSV_PATH)
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

    # Compute row-wise Euclidean distance between the new instance and each centroid.
    # Previously the code called np.linalg.norm on the full matrix which returned a single
    # scalar; that caused the argmin to always pick cluster 0. Use axis=1 to get per-row
    # distances and then select the nearest centroid.
    centroids = df_centroid.iloc[:, 0:12].to_numpy(dtype=float)
    diff = centroids - nueva_instancia.reshape(1, -1)
    distancias = np.linalg.norm(diff, axis=1)
    cluster_predicho = int(np.argmin(distancias))

    return cluster_predicho
