import time

from stqdm import stqdm

import streamlit as st
from utils.constants.categories import *
from utils.constants.streamlit_artifacts import *

#### TABS
ajustes_tab, resultados_tab = st.tabs(("Ajustes", "Resultados"))
with ajustes_tab:
    ############
    # Paciente #
    ############
    st.header("Paciente")

    paciente_column1, paciente_column2, paciente_column3 = st.columns(3)

    with paciente_column1:
        edad: int = st.number_input("Edad", min_value=10, max_value=120, value=10)
        apache: int = st.number_input("Apache", min_value=0, max_value=1000, value=0, help=HELP_MSG_APACHE)
        estad_uti: int = st.number_input("Estadía UTI", min_value=0, max_value=1000, value=0, help=HELP_MSG_ESTAD_UTI)
        tiempo_va: int = st.number_input("Tiempo de VA", min_value=0, max_value=1000, value=0)
    with paciente_column2:
        diagn1: str = st.selectbox("Diagnostico 1", tuple(DIAG_PREUCI.values()), )
        diagn2: str = st.selectbox("Diagnostico 2", tuple(DIAG_PREUCI.values()), )
        estad_preuti: int = st.number_input("Estadía Pre-UTI", min_value=0, max_value=1000, value=0,
                                            help=HELP_MSG_ESTAD_PREUTI)
    with paciente_column3:
        diagn3: str = st.selectbox("Diagnostico 3", tuple(DIAG_PREUCI.values()), )
        diagn4: str = st.selectbox("Diagnostico 4", tuple(DIAG_PREUCI.values()), )
        tipo_va: str = st.selectbox("Tipo de VA", tuple(TIPO_VENT.values()), )

    ##############
    # Simulación #
    ##############
    st.header("Simulación")

    sim_column1, sim_column2, = st.columns(2, gap="small", vertical_alignment="center")

    corridas_simulacion = st.number_input("Corridas de la Simulación", min_value=1, max_value=1000, value=50,
                                          help=HELP_MSG_CORRIDA_SIM)
    with sim_column1:
        boton_comenzar = st.button("Comenzar Simulación", type="primary")
    with sim_column2:
        boton_detener = st.button("Detener Simulación", type="secondary")

    if boton_comenzar:
        for i in stqdm(range(corridas_simulacion), desc="Progreso de la simulación en curso"):
            time.sleep(0.1)
        st.success(f"La simulación ha concluido tras haber completado {corridas_simulacion} iteraciones.")
with resultados_tab:
    st.header("Resultados")

#### SIDEBAR
with st.sidebar:
    st.header("Opciones")
