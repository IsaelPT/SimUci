## Number Input
# Edad
EDAD_MIN = 13
EDAD_MAX = 100
EDAD_DEFAULT = EDAD_MIN

# Apache
APACHE_MIN = 1
APACHE_MAX = 100
APACHE_DEFAULT = 25

# Tiempo VAM
T_VAM_MIN = 1
T_VAM_MAX = 3600
T_VAM_DEFAULT = 100

# Estadía UTI
ESTAD_UTI_MIN = 1
ESTAD_UTI_MAX = 200
ESTAD_UTI_DEFAULT = 25

# Estadía Pre-UTI
ESTAD_PREUTI_MIN = 1
ESTAD_PREUTI_MAX = 200
ESTAD_PREUTI_DEFAULT = 25

# Corridas Simulación
CORRIDAS_SIM_MIN = 20
CORRIDAS_SIM_MAX = 1000
CORRIDAS_SIM_DEFAULT = CORRIDAS_SIM_MIN

# Porciento
PORCIENTO_SIM_MIN = 1
PORCIENTO_SIM_MAX = 100
PORCIENTO_SIM_DEFAULT = 10

# Mensajes de Ayuda en varios Widgets de la aplicación Streamlit
HELP_MSG_APACHE: str = "Valor del APACHE."
HELP_MSG_ESTAD_UTI: str = "Tiempo de estadía en UTI."
HELP_MSG_ESTAD_PREUTI: str = "Tiempo de estadía pre UTI"
HELP_MSG_CORRIDA_SIM: str = "La cantidad de corridas de la simulación brinda un mayor margen de precisión."
HELP_MSG_PORCIENTO_SIM: str = "Índice de porciento para el experiemento."

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
    5: "Causas extrapulmonares"
}

VARIABLES_EXPERIMENTO: list[str] = ["Llegada UCI", "Tiempo Pre VAM", "Comienzo VAM", "Tiempo VAM", "Salida VAM",
                                    "Tiempo Post VAM", "Salida UCI", "Estadia UCI", "Estadia Post UCI", "Egreso"]

RUTA_CSV_DATOS1 = "data\\datos.csv"
RUTA_CSV_DATOS2 = "data\\Ficherodedatos(MO)17-1-2023.csv"
