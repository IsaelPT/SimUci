import numpy as np
import pandas as pd
import scipy.stats as stats
import os

from utils.constants import DFCENTROIDES_CSV_PATH, EXPERIMENT_VARIABLES_FROM_CSV

# Module-level debug switch. Set environment variable UCI_DISTRIB_DEBUG=1 to enable verbose sampling logs.
DEBUG = bool(os.environ.get("UCI_DISTRIB_DEBUG", "0") in ("1", "true", "True"))


# Distribuciones para las variables del cluster 0
def tiemp_VAM_cluster0():
    lambda_dist = 1 / 113.508
    value = stats.expon.rvs(scale=1 / lambda_dist)
    if DEBUG:
        print(f"tiemp_VAM_cluster0: expon scale={1 / lambda_dist:.3f} draw={float(value):.3f}")
    return value


def tiemp_postUCI_cluster0():
    xk = np.array([1, 2, 3])
    pk = np.array([0.6, 0.3, 0.1])

    custom_dist = stats.rv_discrete(name="custom", values=(xk, pk))
    random_numbers = custom_dist.rvs(size=1)

    if 1 in random_numbers:
        value = stats.uniform.rvs(loc=0, scale=168, size=1)
        if DEBUG:
            print(f"tiemp_postUCI_cluster0: branch=1 loc=0 scale=168 draw={float(value):.3f}")
        return value
    elif 2 in random_numbers:
        value = stats.uniform.rvs(loc=192, scale=384, size=1)
        if DEBUG:
            print(f"tiemp_postUCI_cluster0: branch=2 loc=192 scale=384 draw={float(value):.3f}")
        return value
    elif 3 in random_numbers:
        value = stats.uniform.rvs(loc=408, scale=648, size=1)
        if DEBUG:
            print(f"tiemp_postUCI_cluster0: branch=3 loc=408 scale=648 draw={float(value):.3f}")
        return value


def estad_UTI_cluster0():
    forma = 1.37958
    escala = 262.212
    weibull_dist = stats.weibull_min(forma, scale=escala)
    value = weibull_dist.rvs(size=1)
    if DEBUG:
        print(f"estad_UTI_cluster0: weibull shape={forma} scale={escala} draw={float(value):.3f}")
    return value


# Distribuciones para las variables del cluster 1
def tiemp_VAM_cluster1():
    lambda_dist = 1 / 200
    value = stats.expon.rvs(scale=1 / lambda_dist)
    if DEBUG:
        print(f"tiemp_VAM_cluster1: expon scale={1 / lambda_dist:.3f} draw={float(value):.3f}")
    return value


def tiemp_postUCI_clustet1():
    forma = 3.63023
    escala = 1214.29
    weibull_dist = stats.weibull_min(forma, scale=escala)
    value = weibull_dist.rvs(size=1)
    if DEBUG:
        print(f"tiemp_postUCI_clustet1: weibull shape={forma} scale={escala} draw={float(value):.3f}")
    return value


def estad_UTI_cluster1():
    forma = 1.57768
    escala = 472.866
    weibull_dist = stats.weibull_min(forma, scale=escala)
    value = weibull_dist.rvs(size=1)
    if DEBUG:
        print(f"estad_UTI_cluster1: weibull shape={forma} scale={escala} draw={float(value):.3f}")
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
    # Select the first N numeric columns from the centroid file where N is the number
    # of features expected by the clustering function (based on EXPERIMENT_VARIABLES_FROM_CSV).
    try:
        n_features = len(EXPERIMENT_VARIABLES_FROM_CSV)
    except Exception:
        n_features = 11

    # Select numeric columns to be robust to any index or extra columns in the CSV
    numeric_cols = df_centroid.select_dtypes(include=[float, int]).columns.tolist()
    if len(numeric_cols) < n_features:
        # Fallback: use first n_features columns of the raw frame
        centroids = df_centroid.iloc[:, 0:n_features].to_numpy(dtype=float)
    else:
        centroids = df_centroid[numeric_cols[:n_features]].to_numpy(dtype=float)

    feat = nueva_instancia.reshape(1, -1).astype(float)
    # If shapes don't align (defensive), trim or pad
    if centroids.shape[1] != feat.shape[1]:
        minc = min(centroids.shape[1], feat.shape[1])
        centroids = centroids[:, :minc]
        feat = feat[:, :minc]

    diff = centroids - feat
    distancias = np.linalg.norm(diff, axis=1)
    cluster_predicho = int(np.argmin(distancias))
    if DEBUG:
        print(
            f"clustering: features={feat.flatten().tolist()}\ncentroids_shape={centroids.shape}\ndistances={distancias.tolist()} chosen={cluster_predicho}"
        )
        # print chosen centroid row for inspection
        try:
            chosen_row = centroids[cluster_predicho].tolist()
            print(f"clustering: chosen_centroid={chosen_row}")
        except Exception:
            pass

    return cluster_predicho
