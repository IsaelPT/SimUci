from datetime import datetime

import streamlit as st

from st_utils.constants import *
from st_utils.helpers import *
from uci.experiment import *
from uci.stats import *

#### TABS
simulacion_tab, datos_reales_tab, comparaciones_tab = st.tabs(
    ("Simulación", "Datos reales", "Comparaciones")
)

with simulacion_tab:
    ############
    # Paciente #
    ############
    st.header("Paciente")

    # ID Paciente
    # IMPORTANT: EL ID DEL PACIENTE ESTÁ ALMACENADO DENTRO DEL SESSION_STATE!
    if "id_paciente" not in st.session_state:
        st.session_state.id_paciente = generate_id()
    col1_nuevo_paciente, col2_nuevo_paciente = st.columns([1, 1])
    with col1_nuevo_paciente:
        nuevo_paciente = st.button("Nuevo paciente")
        if nuevo_paciente:
            st.session_state.id_paciente = generate_id()
    with col2_nuevo_paciente:
        st.session_state.id_paciente = st.text_input(
            label="ID Paciente",
            value=st.session_state.id_paciente,
            max_chars=10,
            placeholder="ID Paciente",
            label_visibility="collapsed",
        )

    # Ingresar Datos Paciente
    col1_paciente, col2_paciente, col3_paciente = st.columns(3)
    with col1_paciente:
        opcion_edad: int = st.number_input(
            label="Edad", min_value=EDAD_MIN, max_value=EDAD_MAX, value=EDAD_DEFAULT
        )
        opcion_apache: int = st.number_input(
            label="Apache",
            min_value=APACHE_MIN,
            max_value=APACHE_MAX,
            value=APACHE_DEFAULT,
            help=HELP_MSG_APACHE,
        )
        opcion_tiempo_vam: int = st.number_input(
            label="Tiempo en Ventilación Artificial",
            min_value=T_VAM_MIN,
            max_value=T_VAM_MAX,
            value=T_VAM_DEFAULT,
        )
        opcion_estad_uti: int = st.number_input(
            label="Tiempo de Estadía en UCI",
            min_value=ESTAD_UTI_MIN,
            max_value=ESTAD_UTI_MAX,
            value=ESTAD_UTI_DEFAULT,
            help=HELP_MSG_ESTAD_UTI,
        )
    with col2_paciente:
        opcion_insuf_resp: str = st.selectbox(
            label="Tipo de Insuficiencia Respiratoria",
            options=tuple(INSUF_RESP.values()),
            index=1,
        )
        opcion_tipo_vam: str = st.selectbox(
            label="Tipo de Ventilación Artificial",
            options=tuple(TIPO_VENT.values()),
        )
        opcion_estad_preuti: int = st.number_input(
            label="Tiempo de Estadía Pre-UCI",
            min_value=ESTAD_PREUTI_MIN,
            max_value=ESTAD_PREUTI_MAX,
            value=ESTAD_PREUTI_DEFAULT,
            help=HELP_MSG_ESTAD_PREUTI,
        )
        input_porciento = st.number_input(
            label="Porciento de Tiempo en UCI",
            min_value=PORCIENTO_SIM_MIN,
            max_value=PORCIENTO_SIM_MAX,
            value=PORCIENTO_SIM_DEFAULT,
            help=HELP_MSG_PORCIENTO_SIM,
        )
    with col3_paciente:
        opcion_diagn1: str = st.selectbox(
            label="Diagnóstico 1", options=tuple(DIAG_PREUCI.values()), index=0
        )
        opcion_diagn2: str = st.selectbox(
            label="Diagnóstico 2", options=tuple(DIAG_PREUCI.values()), index=0
        )
        opcion_diagn3: str = st.selectbox(
            label="Diagnóstico 3", options=tuple(DIAG_PREUCI.values()), index=0
        )
        opcion_diagn4: str = st.selectbox(
            label="Diagnóstico 4", options=tuple(DIAG_PREUCI.values()), index=0
        )

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

    ##############
    # Simulación #
    ##############
    st.header("Simulación")

    diag_ok = False
    insuf_ok = False
    resultado_experimento = pd.DataFrame()
    if "df_resultado" not in st.session_state:
        st.session_state.df_resultado = (
            pd.DataFrame()
        )  # session_state para visualizar datos simulación.

    sim_buttons_container = st.container()
    with sim_buttons_container:
        corridas_sim = st.number_input(
            "Corridas de la Simulación",
            min_value=CORRIDAS_SIM_MIN,
            max_value=CORRIDAS_SIM_MAX,
            value=CORRIDAS_SIM_DEFAULT,
            help=HELP_MSG_CORRIDA_SIM,
        )
        boton_comenzar = st.button(
            "Comenzar Simulación", type="primary", use_container_width=True
        )

    # Mostrar DataFrame con resultado de la simulación para ese paciente.
    if not st.session_state.df_resultado.empty:
        toggle_format = st.toggle(
            "Tiempo en días",
            value=True,
            help="Formato de tiempo. Si no se selecciona, se mostrará en horas.",
        )
        df_simulacion = build_df_stats(
            data=st.session_state.df_resultado,
            sample_size=corridas_sim,
            include_mean=True,
            include_std=True,
            include_confint=True,
            include_info_label=True,
        )
        if toggle_format:
            df_simulacion = format_df_time(df_simulacion)

        st.dataframe(
            df_simulacion,
            hide_index=True,
            use_container_width=True,
        )

        # Lógica para guardar resultados localmente.
        csv = st.session_state.df_resultado.to_csv(index=False).encode("UTF-8")
        boton_guardar = st.download_button(
            label="Guardar resultados",
            data=csv,
            file_name=f"Experimento-Paciente-ID-{st.session_state.id_paciente}.csv",
            mime="text/csv",
            use_container_width=True,
            key="guardar_simulacion",
        )

        st.success(
            f"La simulación ha concluido tras haber completado {corridas_sim} iteraciones."
        )

    if boton_comenzar:
        # Validación de campos para realizar simulación.
        if not value_is_zero([diagn1, diagn2, diagn3, diagn4]):  # Diagnósticos OK?
            diag_ok = True
        else:
            st.warning(
                f"Todos los diagnósticos están vacíos. Mínimo incluya un diagnóstico para la simulación."
            )
        if not value_is_zero(insuf_resp):  # Insuficiencia Respiratoria OK?
            insuf_ok = True
        else:
            st.warning(f"Seleccione un tipo de Insuficiencia Respiratoria.")

        # Desarrollo de la Simulación.
        if diag_ok and insuf_ok:
            try:
                # Experimento / Simulación.
                resultado_experimento = start_experiment(
                    corridas_sim,
                    edad,
                    diagn1,
                    diagn2,
                    diagn3,
                    diagn4,
                    apache,
                    insuf_resp,
                    insuf_resp,
                    estadia_uti,
                    tiempo_vam,
                    estadia_preuti,
                    input_porciento,
                )

                # Guardar resultados.
                path_base = f"experimentos\\paciente-id-{st.session_state.id_paciente}"
                if not os.path.exists(path_base):
                    os.makedirs(path_base)
                fecha: str = datetime.now().strftime("%d-%m-%Y")
                path: str = (
                    f"{path_base}\\experimento-id {generate_id(5)} fecha {fecha} corridas {corridas_sim}.csv"
                )
                resultado_experimento.to_csv(path, index=False)
                st.session_state.df_resultado = resultado_experimento

                st.rerun()
            except Exception as e:
                st.exception(f"No se pudo efectuar la simulación. Error asociado:\n{e}")

with datos_reales_tab:
    ##############
    # Validación #
    ##############
    st.header("Validación con Datos Reales")

    df_data = pd.read_csv(RUTA_FICHERODEDATOS_CSV)
    df_data.index.name = "Paciente"

    st.markdown(
        "Conjunto de datos que se utilizan para realizar las pruebas de comparaciones.\
            Estos son datos recopilados de pacientes reales en estudios anteriormente realizados."
    )
    html_text = f'<p style="color:{PRIMARY_COLOR};">Puede seleccionar una fila para realizar una simulación al paciente seleccionado.</p>'
    st.markdown(html_text, unsafe_allow_html=True)

    df_selection: int = st.dataframe(
        df_data,
        key="data",
        on_select="rerun",
        selection_mode=["single-row"],
        hide_index=False,
        height=300,
    )["selection"]["rows"]

    # df_selection devuelve un dict de la forma:
    # {
    #     "selection": {
    #         "rows": [0, 1, 2, ...],
    #         "columns": [0, 1, 2, ...]
    #     }
    # }

    df_selection = df_selection[0] if df_selection else None

    # DataFrame Datos Reales
    if df_selection is not None or df_selection == 0:
        st.markdown("Indicadores estadísticos del paciente seleccionado")

        e: tuple[float] = simulate_real_data(RUTA_FICHERODEDATOS_CSV, df_selection)

        # DataFrame con experimento con datos reales
        df_sim_datos_reales = build_df_stats(
            e,
            CORRIDAS_SIM_DEFAULT,
            include_mean=True,
            include_std=True,
            include_confint=True,
            include_info_label=True,
        )
        st.dataframe(
            format_df_time(df_sim_datos_reales),
            hide_index=True,
            use_container_width=True,
        )

    if "df_sim_datos_reales" not in st.session_state:
        st.session_state.df_sim_datos_reales = pd.DataFrame()

    # Simular todos los datos en la tabla.
    if st.button(
        "Simular todos los datos en la tabla",
        type="primary",
        use_container_width=True,
        key="simular_tabla",
    ):
        with st.spinner("Simulando todos los datos en la tabla..."):
            lst_e: list[tuple[float]] = simulate_real_data(
                RUTA_FICHERODEDATOS_CSV, df_selection=None
            )

            # DataFrame con conjunto de todos los resultados de simulaciones a todos los pacientes de la tabla.
            df_sim_datos_reales = build_df_stats(
                lst_e,
                CORRIDAS_SIM_DEFAULT,
                include_mean=True,
                include_std=False,
                include_confint=False,
                include_info_label=False,
            )
            df_sim_datos_reales.index.name = "Paciente"
            st.session_state.df_sim_datos_reales = df_sim_datos_reales

    # Mostrar simulación con datos reales.
    if not st.session_state.df_sim_datos_reales.empty:
        st.dataframe(
            format_df_time(st.session_state.df_sim_datos_reales),
            hide_index=False,
            use_container_width=True,
        )

        # Lógica para guardar resultados localmente.
        csv_sim_datos_reales = st.session_state.df_sim_datos_reales.to_csv(
            index=False
        ).encode("UTF-8")

        st.download_button(
            label="Guardar resultados",
            data=csv_sim_datos_reales,
            file_name=f"Experimentos con Datos Reales.csv",
            mime="text/csv",
            use_container_width=True,
            key="guardar_sim_datos_reales",
        )

        st.success(f"Se ha realizado la simulación para cada paciente correctamente.")

with comparaciones_tab:
    comparacion_wilcoxon = st.expander("Comparación vía Wilcoxon", expanded=True)
    comparacion_friedman = st.expander("Comparación vía Friedman", expanded=True)

    with comparacion_wilcoxon:
        st.header("Prueba de Wilcoxon")

        file_upl1: UploadedFile
        file_upl2: UploadedFile
        df_experimento1 = pd.DataFrame()
        df_experimento2 = pd.DataFrame()

        # File Uploader.
        col1_file_upl, col2_file_upl = st.columns(2)
        with col1_file_upl:
            file_upl1 = st.file_uploader(label="Resultado Experimento 1", type=[".csv"])
            if file_upl1:
                df_experimento1 = bin_to_df(file_upl1)
                if df_experimento1.empty:
                    st.warning("Aún no se han cargado datos del experimento 1.")
                else:
                    st.dataframe(df_experimento1, height=200, hide_index=True)
        with col2_file_upl:
            file_upl2 = st.file_uploader(label="Resultado Experimento 2", type=[".csv"])
            if file_upl2:
                df_experimento2 = bin_to_df(file_upl2)
                if df_experimento2.empty:
                    st.warning("Aún no se han cargado datos del experimento 2.")
                else:
                    st.dataframe(df_experimento2, height=200, hide_index=True)

        # Columna a comparar.
        opcion_col_comparacion = st.selectbox(
            "Escoja una columna para comparar", VARIABLES_EXPERIMENTO, key=1
        )
        boton_comparacion = st.button(
            "Realizar prueba de Wilcoxon",
            type="primary",
            use_container_width=True,
            key=2,
        )

        # Comparación Wilcoxon.
        with st.container():
            if boton_comparacion:
                if df_experimento1.empty or df_experimento2.empty:
                    st.warning(
                        "No se puede realizar la comparación. \
                    Se detectan datos vacíos o falta de datos en los experimentos."
                    )
                else:
                    x: DataFrame = df_experimento1[opcion_col_comparacion]
                    y: DataFrame = df_experimento2[opcion_col_comparacion]
                    if x.equals(y):
                        st.error(
                            'Imposible realizar prueba de Wilcoxon cuando la diferencia entre los elementos \
                        de "x" y "y" es cero para todos los elementos. Verifique que no cargó el mismo \
                        experimento dos veces.'
                        )
                    else:
                        # Corrección de que existen la misma cantidad de filas en ambas tablas.
                        len_info_msg = (
                            lambda exp: f"Se eliminaron filas del experimento {exp} para coincidir \
                                con el experimento {2 if exp == 1 else 1} ({len_dif} filas diferentes)."
                        )
                        len_dif = abs(len(x) - len(y))
                        if (
                            x.shape[0] > y.shape[0]
                        ):  # La cantidad de filas de x, excede las de y.
                            x = x.head(y.shape[0])
                            st.info(len_info_msg(1))
                        elif (
                            y.shape[0] > x.shape[0]
                        ):  # La cantidad de filas de y, excede las de x.
                            y = y.head(x.shape[0])
                            st.info(len_info_msg(2))

                        try:
                            # Test de Wilcoxon
                            wilcoxon_data = Wilcoxon()
                            wilcoxon_data.test(x, y)

                            # Mostrar Resultado
                            df_mostrar = build_df_test_result(
                                statistic=wilcoxon_data.statistic,
                                p_value=wilcoxon_data.p_value,
                            )
                            st.dataframe(
                                df_mostrar, hide_index=True, use_container_width=True
                            )
                        except Exception as e:
                            st.exception(e)

    with comparacion_friedman:
        st.header("Prueba de Friedman")

        file_upl_experimentos: list[UploadedFile]
        dataframes_experimentos: list[DataFrame]

        # File Uploader.
        with st.container():
            file_upl_experimentos = st.file_uploader(
                label="Experimentos", type=[".csv"], accept_multiple_files=True
            )
            dataframes_experimentos = bin_to_df(file_upl_experimentos)

        # Columna a comparar.
        opcion_col_comparacion = st.selectbox(
            "Escoja una columna para comparar", VARIABLES_EXPERIMENTO, key=3
        )
        boton_comparacion = st.button(
            "Realizar prueba de Friedman",
            type="primary",
            use_container_width=True,
            key=4,
        )

        with st.container():
            if boton_comparacion:
                if len(file_upl_experimentos) == 0:
                    st.warning(
                        "No se han cargado datos de resultados de experimentos para realizar la prueba."
                    )
                elif not len(file_upl_experimentos) >= 3:
                    st.warning(
                        "Debe usted cargar más de 3 muestras para realizar la prueba."
                    )
                else:
                    adjusted_sample_tuple = adjust_df_sizes(
                        [df[opcion_col_comparacion] for df in dataframes_experimentos]
                    )

                    samples_selection = adjusted_sample_tuple[0]
                    min_sample_size = adjusted_sample_tuple[1]
                    if min_sample_size != -1:
                        st.info(
                            f"Se eliminaron filas de las tablas de los experimentos para realizar el examen. \
                        Todas las tablas pasaron a tener {min_sample_size} filas."
                        )

                    try:
                        # Test de Friedman.
                        friedman_result = Friedman()
                        friedman_result.test(*samples_selection)

                        # Mostrar Resultado.
                        df_mostrar = build_df_test_result(
                            statistic=friedman_result.statistic,
                            p_value=friedman_result.p_value,
                        )
                        st.dataframe(
                            df_mostrar, hide_index=True, use_container_width=True
                        )
                    except Exception as e:
                        st.exception(e)
