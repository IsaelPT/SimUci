import pandas as pd


def cargar_fichero(path: str, column: str) -> list:
    """
    Retorna una columna organizada por fecha de ingreso

    Args:
        path (str): Direccion del archivo de datos
        column (str): Nombre de la columna a devolver

    Returns:
        list: Una lista de los valores de la columna
    """
    df = pd.read_csv(
        path,
        index_col=0,
        parse_dates=["fecha_ingreso", "fecha_egreso", "fecha_ing_uci", "fecha_egr_uci"],
    )
    df["tiempo_vam"] = df["tiempo_vam"].astype(int)
    df["diagnostico_preuci"] = df["diagnostico_preuci"].astype("category")
    estadia = list(df.sort_values("fecha_ingreso")[column])
    return estadia


def get_fecha_ingreso(path: str):
    """
    Genrador de las fechas de ingreso

    Args:
        path (str): Direccion del archivo de datos

    Yields:
        tupla: Tupla con la fecha siguiente a la que esta y la fecha actual
    """
    fecha_ingreso = cargar_fichero(path, "fecha_ingreso")
    fecha = fecha_ingreso[0]

    for fecha_siguiente in fecha_ingreso:
        yield (fecha_siguiente, fecha)
        fecha = fecha_siguiente


def get_fecha_egreso(path: str):
    """
    Generador de fecha de egreso

    Args:
        path (str): Direccion del archivo de datos

    Yields:
        datetime: La fecha de egreso
    """
    fecha_ingreso = cargar_fichero(path, "fecha_egreso")
    for fecha in fecha_ingreso:
        yield fecha


def get_fecha_ing_uci(path: str):
    """
    Generador de las fechas de ingreso a la UCI

    Args:
        path (str): Direccion del archivo de datos

    Yields:
        datetime: La fecha de ingreso a la UCI
    """
    fecha_ingreso = cargar_fichero(path, "fecha_ing_uci")
    for fecha in fecha_ingreso:
        yield fecha


def get_tiempo_vam(path: str):
    """
    Generador del tiempo en VAM

    Args:
        path (str): Direccion del archivo de datos

    Yields:
        int: Tiempo que esta en VAM
    """
    tiempo_vam = cargar_fichero(path, "tiempo_vam")
    for horas in tiempo_vam:
        yield horas


def get_fecha_egr_uci(path: str):
    """
    Genrador de fecha de egreso de la UCi

    Args:
        path (str): Direccion del archivo de datos

    Yields:
        datetime: La fecha de egreso de la UCI
    """
    fecha_ingreso = cargar_fichero(path, "fecha_egr_uci")
    for fecha in fecha_ingreso:
        yield fecha


def get_estadia_uci(path: str):
    """
    Genreador de la estadia en la UCI

    Args:
        path (str): Direccion del archivo de datos

    Yields:
        int: La cantidad de dias en la UCI
    """
    estadia = cargar_fichero(path, "estadia_uci")
    for est in estadia:
        yield est


def get_sala_egreso(path: str):
    """
    Generador de la sala de egreso de la UCI

    Args:
        path (str): Direccion del archivo de datos

    Yields:
        str: La sala de egreso de la UCI
    """
    salas = cargar_fichero(path, "sala_egreso")
    for sala in salas:
        yield sala


def get_evolucion(path: str):
    """
    Generador de la evolucion del paciente(vive o fallece)

    Args:
        path (str): Direccion del archivo de datos

    Yields:
        str: La evolucion del paciente
    """
    evoluciones = cargar_fichero(path, "evolucion")
    for evolucion in evoluciones:
        yield evolucion


def get_diagnostico(path: str):
    """
    Generador del diagnostico antes de ingresar a la UCI

    Args:
        path (str): Direccion del archivo de datos

    Yields:
        str: Diagnostico del paciente antes de entrar a la UCI
    """
    diagnosticos = cargar_fichero(path, "diagnostico_preuci")
    for daignostico in diagnosticos:
        yield daignostico


def get_diagnostico_list(path: str):
    """
    Obtiene los diagnosticos de los paciente

    Args:
        path (str): Direccion del archivo de datos

    Returns:
        list: Lista de todos los diagnosticos
    """
    df = pd.read_csv(path)
    diagnostico_list = df["diagnostico_preuci"].unique()
    return diagnostico_list


def get_time_simulation(path: str):
    """
    Obtiene la cantidad de tiempo del archivo de datos

    Args:
        path (str): Direccion del archivo de datos

    Returns:
        int: La cantidad de horas que hay en el archivo de datos
    """
    ingreso = cargar_fichero(path, "fecha_ingreso")[0]
    egreso = sorted(cargar_fichero(path, "fecha_egreso"))[-1]
    time_day = egreso - ingreso
    print(time_day)
    time = time_day.days * 24
    return time
