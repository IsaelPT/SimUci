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
CORRIDAS_SIM_MIN = 1
CORRIDAS_SIM_MAX = 1000
CORRIDAS_SIM_DEFAULT = 30

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
