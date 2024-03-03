import pandas as pd

def get_fecha_ingreso(path:str):
    df = pd.read_csv(path, index_col=0, parse_dates=["fecha_ingreso", "fecha_egreso",
                                                      "fecha_ing_uci", "fecha_egr_uci"])
    fecha_ingreso = list(df["fecha_ingreso"].dt.date)
    for fecha in fecha_ingreso:
        yield fecha

def get_fecha_egreso(path:str):
    df = pd.read_csv(path, index_col=0, parse_dates=["fecha_ingreso", "fecha_egreso",
                                                      "fecha_ing_uci", "fecha_egr_uci"])
    fecha_ingreso = list(df["fecha_egreso"].dt.date)
    for fecha in fecha_ingreso:
        yield fecha

def get_fecha_ing_uci(path:str):
    df = pd.read_csv(path, index_col=0, parse_dates=["fecha_ingreso", "fecha_egreso",
                                                      "fecha_ing_uci", "fecha_egr_uci"])
    fecha_ingreso = list(df["fecha_ing_uci"].dt.date)
    for fecha in fecha_ingreso:
        yield fecha

def get_fecha_egr_uci(path:str):
    df = pd.read_csv(path, index_col=0, parse_dates=["fecha_ingreso", "fecha_egreso",
                                                      "fecha_ing_uci", "fecha_egr_uci"])
    fecha_ingreso = list(df["fecha_egr_uci"].dt.date)
    for fecha in fecha_ingreso:
        yield fecha

