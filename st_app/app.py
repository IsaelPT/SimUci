import os.path
from datetime import datetime

import streamlit as st
from pandas import DataFrame
from stqdm import stqdm

from constants import *
from helpers import *
from uci.experiment import *
from utils.categories import *

#### TABS
ajustes_simulacion, resultados_tab = st.tabs(("Ajustes Simulación", "Resultados"))
with ajustes_simulacion:
    ############
    # Paciente #
    ############
    st.header("Paciente")

    paciente_column1, paciente_column2, paciente_column3 = st.columns(3)
    with paciente_column1:
        opcion_edad: int = st.number_input("Edad", min_value=EDAD_MIN, max_value=EDAD_MAX, value=EDAD_DEFAULT)
        opcion_apache: int = st.number_input("Apache", min_value=APACHE_MIN, max_value=APACHE_MAX, value=APACHE_DEFAULT,
                                             help=HELP_MSG_APACHE)
        opcion_tiempo_vam: int = st.number_input("Tiempo de VA", min_value=T_VAM_MIN, max_value=T_VAM_MAX,
                                                 value=T_VAM_DEFAULT)
        opcion_estad_uti: int = st.number_input("Estadía UTI", min_value=ESTAD_UTI_MIN, max_value=ESTAD_UTI_MAX,
                                                value=ESTAD_UTI_DEFAULT, help=HELP_MSG_ESTAD_UTI)
    with paciente_column2:
        opcion_diagn1: str = st.selectbox("Diagnostico 1", tuple(DIAG_PREUCI.values()), )
        opcion_diagn2: str = st.selectbox("Diagnostico 2", tuple(DIAG_PREUCI.values()), )
        opcion_tipo_vam: str = st.selectbox("Tipo de VA", tuple(TIPO_VENT.values()), )
        opcion_estad_preuti: int = st.number_input("Estadía Pre-UTI", min_value=ESTAD_PREUTI_MIN,
                                                   max_value=ESTAD_PREUTI_MAX, value=ESTAD_PREUTI_DEFAULT,
                                                   help=HELP_MSG_ESTAD_PREUTI)
    with paciente_column3:
        opcion_diagn3: str = st.selectbox("Diagnostico 3", tuple(DIAG_PREUCI.values()), )
        opcion_diagn4: str = st.selectbox("Diagnostico 4", tuple(DIAG_PREUCI.values()), )
        opcion_insuf_resp: str = st.selectbox("Tipo Insuficiencia Respiratoria", tuple(INSUF_RESP.values()))

    # Datos Paciente Recolectados (Son los datos de entrada para ser procesados).
    edad: int = opcion_edad
    apache: int = opcion_apache
    diagn1: int = key_categ("diag", opcion_diagn1)
    diagn2: int = key_categ("diag", opcion_diagn2)
    diagn3: int = key_categ("diag", opcion_diagn3)
    diagn4: int = key_categ("diag", opcion_diagn4)
    tipo_vam: int = key_categ("va", opcion_tipo_vam)
    tiempo_vam: int = opcion_tiempo_vam
    estadia_uti: int = opcion_estad_uti
    estadia_preuti: int = opcion_estad_preuti
    insuf_resp: int = key_categ("insuf", opcion_insuf_resp)

    contenedor = st.container()
    with contenedor:
        mostrar_datos = st.checkbox("Mostrar datos del paciente", value=False)
        if mostrar_datos:
            datos_paciente = {
                "Edad": [opcion_edad, None],
                "Diagnóstico 1": [opcion_diagn1, diagn1],
                "Diagnóstico 2": [opcion_diagn2, diagn2],
                "Diagnóstico 3": [opcion_diagn3, diagn3],
                "Diagnóstico 4": [opcion_diagn4, diagn4],
                "Apache": [opcion_apache, None],
                "Tiempo Ventilación Artificial": [opcion_tiempo_vam, None],
                "Tipo Ventilación Artificial": [opcion_tipo_vam, tipo_vam],
                "Estadía UTI": [opcion_estad_uti, None],
                "Estadía Pre-UTI": [opcion_estad_preuti, None],
                "Insuficiencia Respiratoria": [opcion_insuf_resp, insuf_resp],
            }
            df = pd.DataFrame(datos_paciente, index=["Valor", "Índice"]).style.format(precision=0)
            st.dataframe(df)

    ##############
    # Simulación #
    ##############
    st.header("Simulación")

    boton_comenzar = st.button("Comenzar Simulación", type="primary")

    col_corridas, col_porciento = st.columns(2)
    with col_corridas:
        corridas_sim = st.number_input("Corridas de la Simulación", min_value=CORRIDAS_SIM_MIN,
                                       max_value=CORRIDAS_SIM_MAX,
                                       value=CORRIDAS_SIM_DEFAULT, help=HELP_MSG_CORRIDA_SIM)
    with col_porciento:
        porciento = st.number_input("Porciento", min_value=PORCIENTO_SIM_MIN, max_value=PORCIENTO_SIM_MAX,
                                    value=PORCIENTO_SIM_DEFAULT, help=HELP_MSG_PORCIENTO_SIM)

    diag_ok = False
    insuf_ok = False

    if boton_comenzar:
        # Validación de Campos y Valores para realizar simulación.
        if not value_is_zero([diagn1, diagn2, diagn3, diagn4]):
            diag_ok = True
        else:
            st.error(f"Todos los campos de diagnósticos no pueden ser 0 o Vacío.")
        if not value_is_zero(insuf_resp):
            insuf_ok = True
        else:
            st.error(f"Se debe seleccionar un tipo de insuficiencia respiratoria.")

        # Comenzar Simulación si campos están correctos.
        if diag_ok and insuf_ok:
            # Lógica de Guardar resultado.
            fecha = datetime.now().strftime("fecha %d-%m-%Y hora %H-%M-%S")
            ruta_base = f"experimentos\\{corridas_sim} iteraciones, {fecha}"
            if not os.path.exists(ruta_base):
                os.makedirs(ruta_base)

            # Experimento.
            for i in stqdm(range(corridas_sim), desc="Progreso de la simulación en curso"):
                experiment = Experiment(edad, diagn1, diagn2, diagn3, diagn4, apache, insuf_resp,
                                        insuf_resp, estadia_uti, tiempo_vam, estadia_preuti, porciento)
                result: DataFrame = multiple_replication(experiment)
                result.to_csv(f"{ruta_base}\\paciente_ID_{int(id(result))}.csv", index=False)
            st.success(f"La simulación ha concluido tras haber completado {corridas_sim} iteraciones.")
with resultados_tab:
    st.header("Resultados")
