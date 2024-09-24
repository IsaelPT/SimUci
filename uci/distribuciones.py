import scipy.stats as stats
import numpy as np
import pandas as pd

# Distribuciones para las variables del cluster 0
def tiemp_VAM0():

    lambda_dist = 1/113.508
    value = stats.expon.rvs(scale=1/lambda_dist)
    return value

def tiemp_postUCI0():

    xk = np.array([1,2,3])
    pk = np.array([0.6,0.3,0.1])

    custom_dist = stats.rv_discrete(name='custom', values=(xk,pk))
    random_numbers = custom_dist.rvs(size=1)

    if 1 in random_numbers:
        value = stats.uniform.rvs(loc=0,scale=168,size=1)
        return value
    elif 2 in random_numbers:
        value = stats.uniform.rvs(loc=192,scale=384,size=1)
        return value
    elif 3 in random_numbers:
        value = stats.uniform.rvs(loc=408,scale=648,size=1)
        return value

def estad_UTI0():

    forma = 1.37958
    escala = 262.212
    weibull_dist = stats.weibull_min(forma, scale=escala)
    value = weibull_dist.rvs(size=1)
    return value

# Distribuciones para las variables del cluster 1
def tiemp_VAM1():

    lambda_dist = 1/200
    value = stats.expon.rvs(scale=1/lambda_dist)
    return value

def tiemp_postUCI1():

    forma = 3.63023
    escala = 1214.29
    weibull_dist = stats.weibull_min(forma,scale=escala)
    value = weibull_dist.rvs(size=1)
    return value

def estad_UTI1():

    forma = 1.57768
    escala = 472.866
    weibull_dist = stats.weibull_min(forma,scale=escala)
    value = weibull_dist.rvs(size=1)
    return value

# Seleccion de cluster
def clustering(Edad, Diag_Ing1, Diag_Ing2, Diag_Ing3, Diag_Ing4, APACHE, InsufResp, VA, EstadiaUTI, TiempoVAM, Est_PreUCI):
    vag = 0
    if VA == 3 or VA == 2:
        vag = 2
    else:
        vag = 1

    df_centroid = pd.read_csv("uci\DF_Centroides.csv")
    nueva_instancia = np.array([Edad,Diag_Ing1,Diag_Ing2,Diag_Ing3,#
                                Diag_Ing4,APACHE,InsufResp,VA,vag,
                                EstadiaUTI,TiempoVAM,Est_PreUCI])
    distancias = np.linalg.norm(df_centroid.iloc[:, 0:12] - nueva_instancia)
    cluster_predicho = np.argmin(distancias)
    return cluster_predicho
