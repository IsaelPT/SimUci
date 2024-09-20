from time import sleep

from stqdm import stqdm

import streamlit as st
from utils.constants import categories as cons

ajustes_tab, resultados_tab = st.tabs(("Ajustes", "Resultados"))

with ajustes_tab:
    # Paciente
    # --------------------

    st.header("Paciente")
    st.divider()

    column1, column2, column3 = st.columns(3)

    with column1:
        edad: int = st.number_input("Edad", min_value=10, max_value=120, value=10, help="Edad del paciente")
        apache: int = st.number_input("Apache", min_value=0, max_value=1000, value=0, help="Valor del APACHE", )
        estad_uti: int = st.number_input("Estadía UTI", min_value=0, max_value=1000, value=0,
                                         help="Información sobre Estadía UTI")
        tiempo_va = st.number_input("Tiempo de VA", min_value=0, max_value=1000, value=0)
    with column2:
        diagn1: str = st.selectbox("Diagnostico 1", tuple(cons.DIAG_PREUCI.values()), )
        diagn2: str = st.selectbox("Diagnostico 2", tuple(cons.DIAG_PREUCI.values()), )
        estad_preuti: str = st.number_input("Estadía Pre-UTI", min_value=0, max_value=1000, value=0,
                                            help="Información sobre Estadía Pre-UTI")
    with column3:
        diagn3: str = st.selectbox("Diagnostico 3", tuple(cons.DIAG_PREUCI.values()), )
        diagn4: str = st.selectbox("Diagnostico 4", tuple(cons.DIAG_PREUCI.values()), )
        tipo_va: str = st.selectbox("Tipo de VA", tuple(cons.TIPO_VENT.values()), )

    # Simulación
    # --------------

    st.divider()
    st.header("Simulación")

    sim_column1, sim_column2 = st.columns(2, gap="small", vertical_alignment="center")

    # if 'comenzar_clicked' not in st.session_state:
    #     st.session_state.comenzar_clicked = False
    # if 'detener_clicked' not in st.session_state:
    #     st.session_state.detener_clicked = False

    corridas_simulacion = st.number_input("Corridas de la Simulación", min_value=1, max_value=1000, value=50,
                                          help="La cantidad de corridas de la simulación brinda un mayor margen de precisión.")

    with sim_column1:
        boton_comenzar = st.button("Comenzar Simulación", type="primary")
    with sim_column2:
        boton_detener = st.button("Detener Simulación", type="secondary")

    if boton_comenzar:
        for i in stqdm(range(corridas_simulacion), desc="Progreso de la simulación en curso"):
            sleep(0.1)
        st.success(f"La simulación ha concluido tras haber completado {corridas_simulacion} iteraciones.")

with resultados_tab:
    st.header("Resultados")
    st.divider()
