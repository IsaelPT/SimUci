import pandas as pd

def get_fecha_ingreso(path: str):
    df = pd.read_csv(
        path,
        index_col=0,
        parse_dates=["fecha_ingreso", "fecha_egreso", "fecha_ing_uci", "fecha_egr_uci"],
    )

    fecha_ingreso = list(df.sort_values("fecha_ingreso")["fecha_ingreso"])
    fecha = fecha_ingreso[0]
    for fecha_siguiente in fecha_ingreso:
        yield (fecha_siguiente, fecha)
        fecha = fecha_siguiente

def get_fecha_egreso(path: str):
    df = pd.read_csv(
        path,
        index_col=0,
        parse_dates=["fecha_ingreso", "fecha_egreso", "fecha_ing_uci", "fecha_egr_uci"],
    )
    fecha_ingreso = list(df.sort_values("fecha_ingreso")["fecha_egreso"].dt.date)
    for fecha in fecha_ingreso:
        yield fecha

def get_fecha_ing_uci(path: str):
    df = pd.read_csv(
        path,
        index_col=0,
        parse_dates=["fecha_ingreso", "fecha_egreso", "fecha_ing_uci", "fecha_egr_uci"],
    )
    fecha_ingreso = list(df.sort_values("fecha_ingreso")["fecha_ing_uci"].dt.date)
    for fecha in fecha_ingreso:
        yield fecha

def get_tiempo_vam(path:str):
    df = pd.read_csv(path, index_col=0)
    tiempo_vam = list(df.sort_values("fecha_ingreso")["tiempo_vam"])
    for horas in  tiempo_vam:
        yield horas

def get_fecha_egr_uci(path: str):
    df = pd.read_csv(
        path,
        index_col=0,
        parse_dates=["fecha_ingreso", "fecha_egreso", "fecha_ing_uci", "fecha_egr_uci"],
    )
    fecha_ingreso = list(df.sort_values("fecha_ingreso")["fecha_egr_uci"].dt.date)
    for fecha in fecha_ingreso:
        yield fecha

def get_estadia_uci(path: str):
    df = pd.read_csv(
        path,
        index_col=0,
        parse_dates=["fecha_ingreso", "fecha_egreso", "fecha_ing_uci", "fecha_egr_uci"],
    )
    estadia = list(df.sort_values("fecha_ingreso")["estadia_uci"])
    for est in estadia:
        yield est