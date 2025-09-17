from pathlib import Path

## Number Input
# Age
AGE_MIN = 14
AGE_MAX = 100
AGE_DEFAULT = 22

# Apache
APACHE_MIN = 0  # Exceptionally.
APACHE_MAX = 36
APACHE_DEFAULT = 12

# VAM Time
VAM_T_MIN = 24
VAM_T_MAX = 700
VAM_T_DEFAULT = VAM_T_MIN

# UTI Stay
UTI_STAY_MIN = 0
UTI_STAY_MAX = 200
UTI_STAY_DEFAULT = 24

# Pre-UTI stay -> hours
PREUTI_STAY_MIN = 0
PREUTI_STAY_MAX = 34
PREUTI_STAY_DEFAULT = 10

# Simulation runs
SIM_RUNS_MIN = 50
SIM_RUNS_MAX = 100_000
SIM_RUNS_DEFAULT = 200

# Percentage
SIM_PERCENT_MIN = 0
SIM_PERCENT_MAX = 10
SIM_PERCENT_DEFAULT = 3

# Help messages shown in several Streamlit widgets (S)
HELP_MSG_APACHE: str = f"Valor del APACHE (Acute Physiology and Chronic Health Evaluation) es un puntaje clínico que se usa en cuidados intensivos para medir la gravedad de un paciente crítico y estimar su riesgo de mortalidad. Un riesgo bajo sería {APACHE_MIN} y un riesgo alto sería {APACHE_MAX}."
HELP_MSG_UTI_STAY: str = "Tiempo de estadía en Unidad de Terapia Intensiva (UTI) en **horas**."
HELP_MSG_PREUTI_STAY: str = "Tiempo de estadía pre Unidad de Terapia Intensiva (UTI) antes de ingresar a la Unidad de Terapia Intensiva en **horas**."
HELP_MSG_SIM_RUNS: str = "La cantidad de corridas de la simulación. Brinda mayor precisión en los resultado. Una cantidad mayor mejora la precisión, pero también incrementa el tiempo de procesamiento. Una cantidad de 200 corridas es un buen punto de partida para la simulación."
HELP_MSG_SIM_PERCENT: str = "Proporción de tiempo dentro de estancia UCI que se espera antes de entrar en Ventilación."
HELP_MSG_PREDICTION_METRIC: str = "La predicción de fallecimiento del paciente se realiza a través de un modelo de Inteligencia Artificial entrenado con datos de pacientes en Unidades de Cuidados Intensivos. Variables como *Diagnóstico Ingreso 1*, *Diagnóstico Ingreso 2*, *Diagnóstico Egreso 2*, *Tiempo en VAM*, *Apache* y la *Edad* del paciente intervienen en la estimación de probabilidad del modelo."
HELP_MSG_TIME_FORMAT: str = "Formato de tiempo. Activar esta opción muestra los días convertidos en horas."
INFO_STATISTIC: str = "***Statistic***: Este número indica cuánto difieren los datos entre sí, basándose en el orden de las diferencias; un valor más pequeño sugiere que hay más diferencias entre los grupos que se están comparando."
INFO_P_VALUE: str = "***Valor de P***: Este número dice qué tan probable es que las diferencias que ves se deban al azar; si es menor a 0.05, es probable que las diferencias sean reales y no casuales."
HELP_MSG_VAM_TIME: str = "Tiempo en Ventilación Asistida Mecánica (VAM) en **horas**."
VENTILATION_TYPE: dict[int, str] = {0: "Tubo endotraqueal", 1: "Traqueostomía", 2: "Ambas"}
LABEL_TIME_FORMAT = "Tiempo en días"
LABEL_PREDICTION_METRIC = "Predicción de fallecimiento del paciente seleccionado"


PREUCI_DIAG: dict[int, str] = {
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

RESP_INSUF: dict[int, str] = {
    0: "Vacío",
    1: "Respiratorias",
    2: "TCE",
    3: "Estatus posoperatorio",
    4: "Afecciones no traumáticas del SNC",
    5: "Causas extrapulmonares",
}

EXPERIMENT_VARIABLES = ["Tiempo Pre VAM", "Tiempo VAM", "Tiempo Post VAM", "Estadia UCI", "Estadia Post UCI"]

EXPERIMENT_VARIABLES_DATAFRAME = EXPERIMENT_VARIABLES + ["Promedio Predicción"]

# PREDICTION_VARIABLES = []

try:
    CSV_DATA_PATH = Path("data") / "datos.csv"
    FICHERODEDATOS_CSV_PATH = Path("data") / "Ficherodedatos(MO)17-1-2023.csv"
    DFCENTROIDES_CSV_PATH = Path("data") / "DF_Centroides.csv"
    PREDICTIONS_CSV_PATH = Path("data") / "data_with_pred_and_prob.csv"
    PREDICTION_MODEL_PATH = Path("models") / "prediction_model.joblib"
except Exception as experimento:
    print(f"Error loading database files. Exception: {experimento}")

try:
    import toml

    current_dir = Path(__file__).parent
    config_path = current_dir / ".." / ".streamlit" / "config.toml"
    with open(config_path, "r") as f:
        config = toml.load(f)
        PRIMARY_COLOR = config["theme"]["primaryColor"]
        SECUNDARY_BACKGROUND_COLOR = config["theme"]["secondaryBackgroundColor"]
except FileNotFoundError as fnf:
    print(f"Streamlit config file not found: {fnf}")
