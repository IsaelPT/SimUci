import os.path

## Number Input
# Edad
EDAD_MIN = 14
EDAD_MAX = 100
EDAD_DEFAULT = 22

# Apache
APACHE_MIN = 0  # Excepcionalmente.
APACHE_MAX = 36
APACHE_DEFAULT = 5

# Tiempo VAM
T_VAM_MIN = 24
T_VAM_MAX = 700
T_VAM_DEFAULT = T_VAM_MIN

# Estadía UTI
ESTAD_UTI_MIN = 0
ESTAD_UTI_MAX = 200
ESTAD_UTI_DEFAULT = 24

# Estadía Pre-UTI -> días
ESTAD_PREUTI_MIN = 0
ESTAD_PREUTI_MAX = 34
ESTAD_PREUTI_DEFAULT = 10

# Corridas Simulación
CORRIDAS_SIM_MIN = 50
CORRIDAS_SIM_MAX = 1000
CORRIDAS_SIM_DEFAULT = CORRIDAS_SIM_MIN

# Porciento
PORCIENTO_SIM_MIN = 0
PORCIENTO_SIM_MAX = 10
PORCIENTO_SIM_DEFAULT = 2

# Mensajes de Ayuda en varios Widgets de la aplicación Streamlit
HELP_MSG_APACHE: str = "Valor del APACHE."
HELP_MSG_ESTAD_UTI: str = (
    "Tiempo de estadía en UTI (Unidad de Terapia Intensiva) en horas."
)
HELP_MSG_ESTAD_PREUTI: str = "Tiempo de estadía pre UTI (antes de ingresar a la Unidad de Terapia Intensiva) en horas."
HELP_MSG_CORRIDA_SIM: str = (
    "La cantidad de corridas de la simulación brinda un mayor margen de precisión."
)
HELP_MSG_PORCIENTO_SIM: str = "Proporción de tiempo dentro de estancia UCI que se espera antes de entrar en Ventilación."
INFO_STATISTIC: str = "***Statistic***: Este número indica cuánto difieren los datos entre sí, basándose en el orden de las diferencias; un valor más pequeño sugiere que hay más diferencias entre los grupos que estás comparando."
INFO_P_VALUE: str = "***Valor de P***: Este número dice qué tan probable es que las diferencias que ves se deban al azar; si es menor a 0.05, es probable que las diferencias sean reales y no casuales."
HELP_MSG_TIEMPO_VAM: str = "Tiempo en Ventilación Asistida Mecánica (VAM) en horas."
TIPO_VENT: dict[int, str] = {0: "Tubo endotraqueal", 1: "Traqueostomía", 2: "Ambas"}

DIAG_PREUCI: dict[int, str] = {
    0: "Vacío",
    1: "Intoxicación exógena",
    2: "Coma",
    3: "Trauma craneoencefálico severo",
    4: "SPO de toracotomía",
    5: "SPO de laparotomía",
    6: "SPO de amputación",
    7: "SPO de neurología",
    8: "PCR recuperado",
    9: "Encefalopatía metabólica",
    10: "Encefalopatía hipóxica",
    11: "Ahorcamiento incompleto",
    12: "Insuficiencia cardiaca descompensada",
    13: "Obstétrica grave",
    14: "EPOC descompensada",
    15: "ARDS",
    16: "BNB-EH",
    17: "BNB-IH",
    18: "BNV",
    19: "Miocarditis",
    20: "Leptospirosis",
    21: "Sepsis grave",
    22: "DMO",
    23: "Shock séptico",
    24: "Shock hipovolémico",
    25: "Shock cardiogénico",
    26: "IMA",
    27: "Politraumatizado",
    28: "Crisis miasténica",
    29: "Emergencia hipertensiva",
    30: "Status asmático",
    31: "Status epiléptico",
    32: "Pancreatitis",
    33: "Embolismo graso",
    34: "Accidente cerebrovascular",
    35: "Síndrome de apnea del sueño",
    36: "Sangramiento digestivo",
    37: "Insuficiencia renal crónica",
    38: "Insuficiencia renal aguda",
    39: "Trasplante renal",
    40: "Guillain Barré",
}

INSUF_RESP: dict[int, str] = {
    0: "Vacío",
    1: "Respiratorias",
    2: "TCE",
    3: "Estatus posoperatorio",
    4: "Afecciones no traumáticas del SNC",
    5: "Causas extrapulmonares",
}

VARIABLES_EXPERIMENTO = [
    "Tiempo Pre VAM",
    "Tiempo VAM",
    "Tiempo Post VAM",
    "Estadia UCI",
    "Estadia Post UCI",
]

try:
    RUTA_DATOS_CSV = os.path.join("data", "datos.csv")
    RUTA_FICHERODEDATOS_CSV = os.path.join("data", "Ficherodedatos(MO)17-1-2023.csv")
    RUTA_DFCENTROIDES_CSV = os.path.join("data", "DF_Centroides.csv")
    RUTA_PREDICCIONES_CSV = os.path.join("data", "data_with_pred_and_prob.csv")
    RUTA_MODELO_PREDICCION = os.path.join("models", "new_workflow.joblib")
except Exception as experimento:
    print(
        f"Error al cargar el archivo la base de datos.\n>>>\nExcepcion\n>>>{experimento}"
    )

try:
    import toml

    current_dir = os.path.dirname(__file__)
    config_path = os.path.join(current_dir, "..", ".streamlit", "config.toml")
    with open(config_path, "r") as f:
        config = toml.load(f)
        PRIMARY_COLOR = config["theme"]["primaryColor"]
        SECUNDARY_BACKGROUND_COLOR = config["theme"]["secondaryBackgroundColor"]
except FileNotFoundError as fnf:
    print(f"No se encontró el de config.\n{fnf}")
