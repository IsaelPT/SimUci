import os.path
from datetime import datetime

import streamlit as st
from pandas.io.formats.style import Styler
from scipy.stats import wilcoxon

from constants import *
from experiment import *
from helpers import *


#### FUNCTIONS
def build_data_to_show(style: bool = True, show_index: bool = True) -> DataFrame | Styler:
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
    df = pd.DataFrame(datos_paciente)

    # Options
    if show_index:
        index = ["Valor", "Índice"]
        df.index = index
    else:
        df = df.drop(df.index[-1])  # No mostrar la última fila.
    if style:
        return df.style.format(precision=2)
    return df


#### TABS
simulacion_tab, comparaciones_tab, db_tab = st.tabs(("Simulación", "Comparaciones", "Datos"))
with simulacion_tab:
    ############
    # Paciente #
    ############
    st.header("Paciente")

    # ID Paciente
    # WARNING: EL ID DEL PACIENTE ESTÁ ALMACENADO DENTRO DEL SESSION_STATE!
    if "id_paciente" not in st.session_state:
        st.session_state.id_paciente = generate_id()
    col1_nuevo_paciente, col2_nuevo_paciente, col3_nuevo_paciente = st.columns([1, 1, 1])
    with col1_nuevo_paciente:
        nuevo_paciente = st.button("Nuevo paciente")
        if nuevo_paciente:
            st.session_state.id_paciente = generate_id()
    with col2_nuevo_paciente:
        st.session_state.id_paciente = st.text_input(label="ID Paciente", value=st.session_state.id_paciente,
                                                     max_chars=10, placeholder="ID Paciente",
                                                     label_visibility="collapsed")
    # st.caption(f"ID Paciente: {st.session_state.id_paciente}")
    with col3_nuevo_paciente:
        usar_score: bool = st.toggle("Utilizar Score pronóstico")

    # Ingresar Datos Paciente
    paciente_column1, paciente_column2, paciente_column3 = st.columns(3)
    with paciente_column1:
        opcion_edad: int = st.number_input("Edad", min_value=EDAD_MIN, max_value=EDAD_MAX, value=EDAD_DEFAULT)
        opcion_apache: int = st.number_input("Apache", min_value=APACHE_MIN, max_value=APACHE_MAX,
                                             value=APACHE_DEFAULT,
                                             help=HELP_MSG_APACHE)
        opcion_tiempo_vam: int = st.number_input("Tiempo de VA", min_value=T_VAM_MIN, max_value=T_VAM_MAX,
                                                 value=T_VAM_DEFAULT)
        opcion_estad_uti: int = st.number_input("Estadía UTI", min_value=ESTAD_UTI_MIN, max_value=ESTAD_UTI_MAX,
                                                value=ESTAD_UTI_DEFAULT, help=HELP_MSG_ESTAD_UTI)
    with paciente_column2:
        opcion_diagn1: str = st.selectbox("Diagnostico 1", tuple(DIAG_PREUCI.values()), index=1)
        opcion_diagn2: str = st.selectbox("Diagnostico 2", tuple(DIAG_PREUCI.values()), )
        opcion_tipo_vam: str = st.selectbox("Tipo de VA", tuple(TIPO_VENT.values()), )
        opcion_estad_preuti: int = st.number_input("Estadía Pre-UTI", min_value=ESTAD_PREUTI_MIN,
                                                   max_value=ESTAD_PREUTI_MAX, value=ESTAD_PREUTI_DEFAULT,
                                                   help=HELP_MSG_ESTAD_PREUTI)
    with paciente_column3:
        opcion_diagn3: str = st.selectbox("Diagnostico 3", tuple(DIAG_PREUCI.values()), )
        opcion_diagn4: str = st.selectbox("Diagnostico 4", tuple(DIAG_PREUCI.values()), )
        opcion_insuf_resp: str = st.selectbox("Tipo Insuficiencia Respiratoria", tuple(INSUF_RESP.values()),
                                              index=1)
        porciento = st.number_input("Porciento", min_value=PORCIENTO_SIM_MIN, max_value=PORCIENTO_SIM_MAX,
                                    value=PORCIENTO_SIM_DEFAULT, help=HELP_MSG_PORCIENTO_SIM)

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

    # Mostrar Datos Paciente
    mostrar_datos: bool = st.expander("Mostrar datos del paciente", expanded=False)
    with mostrar_datos:
        toggle_precision = st.toggle("Datos precisos", value=False)
        if toggle_precision:
            st.dataframe(build_data_to_show(style=True))
        else:
            st.dataframe(build_data_to_show(style=False))

    ##############
    # Simulación #
    ##############
    st.header("Simulación")

    diag_ok = False
    insuf_ok = False
    resultado_experimento = pd.DataFrame()
    if "df_resultado" not in st.session_state:
        st.session_state.df_resultado = pd.DataFrame()  # session_state para visualizar datos simulación.

    boton_comenzar = st.button("Comenzar Simulación", type="primary", use_container_width=True)
    corridas_sim = st.number_input("Corridas de la Simulación", min_value=CORRIDAS_SIM_MIN,
                                   max_value=CORRIDAS_SIM_MAX, value=CORRIDAS_SIM_DEFAULT,
                                   help=HELP_MSG_CORRIDA_SIM)
    # Visualizar DataFrame con resultado de la simulación para este paciente.
    if not st.session_state.df_resultado.empty:
        toggle_fmt = st.toggle("Tabla con formato", value=True)
        st.dataframe(format_df(st.session_state.df_resultado, toggle_fmt), height=300)

        # Lógica para guardar resultados localmente.
        csv = st.session_state.df_resultado.to_csv(index=False).encode("UTF-8")
        boton_guardar = st.download_button(
            label="Guardar resultados",
            data=csv,
            file_name=f"Experimento-Paciente-ID-{st.session_state.id_paciente}.csv",
            mime="text/csv",
            use_container_width=True
        )

    if boton_comenzar:
        # Validación de campos para realizar simulación.
        if not value_is_zero([diagn1, diagn2, diagn3, diagn4]):  # Diagnósticos OK?
            diag_ok = True
        else:
            st.warning(f"Todos los campos de diagnósticos no pueden ser 0 o Vacío.")
        if not value_is_zero(insuf_resp):  # Insuficiencia Respiratoria OK?
            insuf_ok = True
        else:
            st.warning(f"Se debe seleccionar un tipo de insuficiencia respiratoria.")

        # Desarrollo de Simulación.
        if diag_ok and insuf_ok:
            try:
                # experiment = Experiment(edad, diagn1, diagn2, diagn3, diagn4, apache, insuf_resp,
                #                         insuf_resp, estadia_uti, tiempo_vam, estadia_preuti, porciento)
                # resultado_experimento = multiple_replication(experiment, corridas_sim)

                # Experimento / Simulación.
                resultado_experimento = start_experiment(corridas_sim, edad, diagn1, diagn2, diagn3, diagn4, apache,
                                                         insuf_resp, insuf_resp, estadia_uti, tiempo_vam,
                                                         estadia_preuti, porciento)

                # Guardar resultados.
                path_base = f"experimentos\\paciente-id-{st.session_state.id_paciente}"
                if not os.path.exists(path_base):
                    os.makedirs(path_base)
                fecha: str = datetime.now().strftime('%d-%m-%Y')
                path: str = f"{path_base}\\experimento-id {generate_id(5)} fecha {fecha} corridas {corridas_sim}.csv"
                resultado_experimento.to_csv(path, index=False)
                st.session_state.df_resultado = resultado_experimento

                st.success(f"La simulación ha concluido tras haber completado {corridas_sim} iteraciones.")
                st.rerun()
            except Exception as e:
                st.exception(f"No se pudo efectuar la simulación.\n{e}")

with comparaciones_tab:
    comparaciones_wilcoxon = st.expander("Comparación vía Wilcoxon", expanded=True)
    comparaciones_friedman = st.expander("Comparación vía Friedman", expanded=True)

    with comparaciones_wilcoxon:
        st.header("Prueba de Wilcoxon")

        # Considerando que los datos reales forman parte de una base de datos.
        # En la etapa actual, dicha base de datos está en formato .csv, que queda a despensas de un
        # futuro a implementar como una base de datos SQL o No SQL; incierto ahora mismo.

        data = pd.read_csv(RUTA_FICHERODEDATOS_CSV)
        CAPTION_MSG_REAL_DATA: str = "Datos reales obtenidos de la Base de Datos"

        experimento: UploadedFile
        df_experimento_sim: DataFrame = pd.DataFrame()
        df_experimento_real: DataFrame = pd.DataFrame()

        toggle_if_simulation_with_data = st.toggle("Realizar con datos de simulación", value=True)
        seleccion_fila = st.number_input("Selecciona una fila", min_value=1, max_value=data.shape[0] - 1, value=1)

        r_data: tuple = get_real_data(RUTA_FICHERODEDATOS_CSV, fila_seleccion=seleccion_fila)

        st.caption("Datos seleccionados")
        st.dataframe(data.iloc[[seleccion_fila]])

        if toggle_if_simulation_with_data:
            if not st.session_state.df_resultado.empty:
                df_experimento_sim = st.session_state.df_resultado
                with st.container():
                    col1_wilcoxon, col2_wilcoxon = st.columns(2)
                    with col1_wilcoxon:
                        st.caption(CAPTION_MSG_REAL_DATA)
                        e = start_experiment(
                            corridas_sim,
                            edad=r_data[0],
                            d1=r_data[1],
                            d2=r_data[2],
                            d3=r_data[3],
                            d4=r_data[4],
                            apache=r_data[5],
                            insuf_resp=r_data[6],
                            va=r_data[7],
                            t_vam=r_data[8],
                            est_uti=r_data[9],
                            est_preuti=r_data[10],
                            porciento=porciento
                        )
                        df_experimento_real = e.mean()
                        st.dataframe(df_experimento_real, use_container_width=True)
                        st.write(f"Cantidad de filas {df_experimento_real.shape[0]}")
                    with col2_wilcoxon:
                        st.caption("Media de datos resultantes de la Simulación")
                        df_experimento_sim = pd.DataFrame(df_experimento_sim.mean(), columns=["Valor"])
                        df_experimento_sim.index.name = "Datos"
                        st.dataframe(df_experimento_sim, use_container_width=True)
                        st.write(f"Cantidad de filas {df_experimento_sim.shape[0]}")
            else:
                st.error("No se ha realizado ninguna simulación hasta el momento. \
            Diríjase al apartado \"Simulación\".")
        else:
            # Escoger un archivo donde estén datos de simulación de un paciente.
            col1_r_data, col2_file_upl = st.columns(2)
            with col1_r_data:
                st.caption(CAPTION_MSG_REAL_DATA)
                st.dataframe(r_data, use_container_width=True, height=200)
            with col2_file_upl:
                experimento = st.file_uploader("Datos de resultado de Simulación")
                if experimento:
                    df_experimento_sim = bin_to_df(experimento)
                    if not df_experimento_sim.empty:
                        st.dataframe(df_experimento_sim, height=200)

        # Selección de columna para comparación
        opcion_columna_comparacion = st.selectbox("Escoja una columna para comparar", VARIABLES_EXPERIMENTO, key=1)
        boton_comparacion = st.button("Realizar prueba", type="primary", use_container_width=True, key=2)
        if boton_comparacion:
            if not df_experimento_sim.empty:
                x: DataFrame = df_experimento_real[opcion_columna_comparacion]
                y: DataFrame = df_experimento_sim.loc[opcion_columna_comparacion]
                resultado = wilcoxon(x, y)
                st.write(resultado)
            else:
                st.warning("No se han cargado datos del experimento aún.")

    with comparaciones_friedman:
        st.header("Prueba de Friedman")
        # TEXTO = "La *prueba de Friedman* se usa para comparar dos o más grupos de datos relacionados. \
        # Útil cuando los datos no siguen una distribución normal."
        # st.markdown(TEXTO)

        experimento1: UploadedFile
        experimento2: UploadedFile
        df_experimento1: DataFrame = pd.DataFrame()
        df_experimento2: DataFrame = pd.DataFrame()

        # Subir datos con File Uploader.
        col1_file_upl, col2_file_upl = st.columns(2)
        with col1_file_upl:
            experimento1 = st.file_uploader("Resultado Experimento 1")
            if experimento1:
                df_experimento1 = bin_to_df(experimento1)
                if not df_experimento1.empty:
                    st.dataframe(df_experimento1, height=200)
                    st.write(f"Cantidad de Filas: {df_experimento1.shape[0]}")
                else:
                    st.warning("No se han cargado datos del experimento 1 aún.")
        with col2_file_upl:
            experimento2 = st.file_uploader("Resultado Experimento 2")
            if experimento2:
                df_experimento2 = bin_to_df(experimento2)
                if not df_experimento2.empty:
                    st.dataframe(df_experimento2, height=200)
                    st.write(f"Cantidad de Filas: {df_experimento2.shape[0]}")
                else:
                    st.warning("No se han cargado datos del experimento 2 aún.")

        # Selección de columna para comparación
        opcion_columna_comparacion = st.selectbox("Escoja una columna para comparar", VARIABLES_EXPERIMENTO, key=3)
        boton_comparacion = st.button("Realizar prueba", type="primary", use_container_width=True, key=4)

        # Comparación
        if boton_comparacion:
            if not df_experimento1.empty and not df_experimento2.empty:
                x: DataFrame = df_experimento1[opcion_columna_comparacion]
                y: DataFrame = df_experimento2[opcion_columna_comparacion]
                if not x.equals(y):
                    # Verificar que ambos dataframes tengan la misma cantidad de filas para realizar Wilcoxon.
                    len_dif = abs(len(x) - len(y))
                    len_warning_msg = lambda \
                            exp: f"Se eliminaron filas del experimento {exp} para coincidir \
                        con el experimento {2 if exp == 1 else 1} ({len_dif} filas diferentes)."
                    if x.shape[0] > y.shape[0]:  # La cantidad de filas de x, excede las de y.
                        x = x.head(y.shape[0])
                        st.warning(len_warning_msg(1))
                    elif y.shape[0] > x.shape[0]:  # La cantidad de filas de y, excede las de x.
                        y = y.head(x.shape[0])
                        st.warning(len_warning_msg(2))
                    wilcoxon_data = (x, y)
                    resultado = wilcoxon(x, y)
                    st.write(resultado)
                else:
                    st.error("Imposible realizar prueba de Wilcoxon cuando la diferencia entre los \
                elementos de \"x\" y \"y\" es cero para todos los elementos.")
            else:
                st.warning("No se puede realizar la comparación. \
            Se detectan datos vacíos o falta de datos en los experimentos.")

with db_tab:
    st.markdown("Este es el conjunto de datos que se utilizan para realizar las pruebas de comparaciones.\
            Son datos que han sido recopilados de pacientes reales en estudios anteriores realizados.")
    st.dataframe(pd.read_csv(RUTA_FICHERODEDATOS_CSV))
