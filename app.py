from datetime import datetime

from st_utils.constants import *
from st_utils.helpers import *
from uci.experiment import *
from uci.stats import *

# from scipy.stats import wilcoxon, friedmanchisquare

#### TABS
simulacion_tab, validacion_tab, comparaciones_tab = st.tabs(("Simulación", "Datos reales", "Comparaciones"))

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
        st.session_state.id_paciente = st.text_input(label="ID Paciente", value=st.session_state.id_paciente,
                                                     max_chars=10, placeholder="ID Paciente",
                                                     label_visibility="collapsed")

    # Ingresar Datos Paciente
    col1_paciente, col2_paciente, col3_paciente = st.columns(3)
    with col1_paciente:
        opcion_edad: int = st.number_input("Edad", min_value=EDAD_MIN, max_value=EDAD_MAX, value=EDAD_DEFAULT)
        opcion_apache: int = st.number_input("Apache", min_value=APACHE_MIN, max_value=APACHE_MAX,
                                             value=APACHE_DEFAULT,
                                             help=HELP_MSG_APACHE)
        opcion_tiempo_vam: int = st.number_input("Tiempo de Ventilación Artificial", min_value=T_VAM_MIN,
                                                 max_value=T_VAM_MAX,
                                                 value=T_VAM_DEFAULT)
        opcion_estad_uti: int = st.number_input("Estadía UCI", min_value=ESTAD_UTI_MIN, max_value=ESTAD_UTI_MAX,
                                                value=ESTAD_UTI_DEFAULT, help=HELP_MSG_ESTAD_UTI)
    with col2_paciente:
        opcion_insuf_resp: str = st.selectbox("Tipo Insuficiencia Respiratoria", tuple(INSUF_RESP.values()),
                                              index=1)
        opcion_tipo_vam: str = st.selectbox("Tipo de Ventilación Artificial", tuple(TIPO_VENT.values()), )
        opcion_estad_preuti: int = st.number_input("Estadía Pre-UCI", min_value=ESTAD_PREUTI_MIN,
                                                   max_value=ESTAD_PREUTI_MAX, value=ESTAD_PREUTI_DEFAULT,
                                                   help=HELP_MSG_ESTAD_PREUTI)
        porciento = st.number_input("Porciento Tiempo UCI", min_value=PORCIENTO_SIM_MIN, max_value=PORCIENTO_SIM_MAX,
                                    value=PORCIENTO_SIM_DEFAULT, help=HELP_MSG_PORCIENTO_SIM)
    with col3_paciente:
        opcion_diagn1: str = st.selectbox("Diagnóstico 1", tuple(DIAG_PREUCI.values()), index=0)
        opcion_diagn2: str = st.selectbox("Diagnóstico 2", tuple(DIAG_PREUCI.values()), index=0)
        opcion_diagn3: str = st.selectbox("Diagnóstico 3", tuple(DIAG_PREUCI.values()), index=0)
        opcion_diagn4: str = st.selectbox("Diagnóstico 4", tuple(DIAG_PREUCI.values()), index=0)

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
        st.session_state.df_resultado = pd.DataFrame()  # session_state para visualizar datos simulación.

    sim_buttons_container = st.container()
    with sim_buttons_container:
        corridas_sim = st.number_input("Corridas de la Simulación", min_value=CORRIDAS_SIM_MIN,
                                       max_value=CORRIDAS_SIM_MAX, value=CORRIDAS_SIM_DEFAULT,
                                       help=HELP_MSG_CORRIDA_SIM)
        boton_comenzar = st.button("Comenzar Simulación", type="primary", use_container_width=True)

    # Visualizar DataFrame con resultado de la simulación para este paciente.
    if not st.session_state.df_resultado.empty:
        toggle_fmt = st.toggle("Tabla con formato", value=True)
        st.dataframe(
            format_df_time(st.session_state.df_resultado, toggle_fmt),
            hide_index=True,
            use_container_width=True,
            height=300
        )

        # Lógica para guardar resultados localmente.
        csv = st.session_state.df_resultado.to_csv(index=False).encode("UTF-8")
        boton_guardar = st.download_button(
            label="Guardar resultados",
            data=csv,
            file_name=f"Experimento-Paciente-ID-{st.session_state.id_paciente}.csv",
            mime="text/csv",
            use_container_width=True
        )

        st.success(f"La simulación ha concluido tras haber completado {corridas_sim} iteraciones.")

    if boton_comenzar:
        # Validación de campos para realizar simulación.
        if not value_is_zero([diagn1, diagn2, diagn3, diagn4]):  # Diagnósticos OK?
            diag_ok = True
        else:
            st.warning(f"Todos los diagnósticos están vacíos. Mínimo incluya un diagnóstico para la simulación.")
        if not value_is_zero(insuf_resp):  # Insuficiencia Respiratoria OK?
            insuf_ok = True
        else:
            st.warning(f"Seleccione un tipo de Insuficiencia Respiratoria.")

        # Desarrollo de la Simulación.
        if diag_ok and insuf_ok:
            try:
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

                st.rerun()
            except Exception as e:
                st.exception(f"No se pudo efectuar la simulación. Error asociado:\n{e}")

with validacion_tab:
    st.markdown("Este es el conjunto de datos que se utilizan para realizar las pruebas de comparaciones.\
            Son datos que han sido recopilados de pacientes reales en estudios anteriores realizados.")

    df_data = pd.read_csv(RUTA_FICHERODEDATOS_CSV)

    selection: list[int] = st.dataframe(
        df_data,
        key="data",
        on_select="rerun",
        selection_mode=["multi-row"],
        hide_index=True,
        height=300
    )["selection"]["rows"]  # dict -> {"selection": {"rows": [0, 1, 2, ...], "columns": []}}

    # DataFrame Datos Reales
    if len(selection) >= 2:
        df_real_data: DataFrame = get_real_data(RUTA_FICHERODEDATOS_CSV, row_selection=selection)
        with st.container():
            st.markdown("Datos Reales seleccionados")
            res = build_df_stats(df_real_data)
            st.dataframe(
                res,
                hide_index=True,
                use_container_width=True,
            )
    # DataFrame Simulacion
    if not st.session_state.df_resultado.empty:  # Hacemos referencias a los resultados de la Simulación
        df_experimento_sim = st.session_state.df_resultado
        sample_size = df_experimento_sim.shape[0]
        with st.container():
            st.markdown("Datos resultantes de la Simulación")
            res = build_df_stats(df_experimento_sim)
            st.dataframe(
                res,
                hide_index=True,
                use_container_width=True,
            )
    else:
        st.error("No se ha realizado ninguna simulación hasta el momento. Diríjase al apartado \"Simulación\".")

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
                    st.warning("No se han cargado datos del experimento 1 aún.")
                else:
                    st.dataframe(df_experimento1, height=200, hide_index=True)
        with col2_file_upl:
            file_upl2 = st.file_uploader(label="Resultado Experimento 2", type=[".csv"])
            if file_upl2:
                df_experimento2 = bin_to_df(file_upl2)
                if df_experimento2.empty:
                    st.warning("No se han cargado datos del experimento 2 aún.")
                else:
                    st.dataframe(df_experimento2, height=200, hide_index=True)

        # Columna a comparar.
        opcion_col_comparacion = st.selectbox("Escoja una columna para comparar", VARIABLES_EXPERIMENTO, key=1)
        boton_comparacion = st.button("Realizar prueba de Wilcoxon", type="primary", use_container_width=True, key=2)

        # Comparación.
        with st.container():
            if boton_comparacion:
                if df_experimento1.empty or df_experimento2.empty:
                    st.warning("No se puede realizar la comparación. \
                    Se detectan datos vacíos o falta de datos en los experimentos.")
                else:
                    x: DataFrame = df_experimento1[opcion_col_comparacion]
                    y: DataFrame = df_experimento2[opcion_col_comparacion]
                    if x.equals(y):
                        st.error("Imposible realizar prueba de Wilcoxon cuando la diferencia entre los elementos \
                        de \"x\" y \"y\" es cero para todos los elementos. Verifique que no cargó el mismo \
                        experimento dos veces.")
                    else:
                        # Corrección de que existen la misma cantidad de filas en ambas tablas.
                        len_info_msg = lambda \
                                exp: f"Se eliminaron filas del experimento {exp} para coincidir \
                                con el experimento {2 if exp == 1 else 1} ({len_dif} filas diferentes)."
                        len_dif = abs(len(x) - len(y))
                        if x.shape[0] > y.shape[0]:  # La cantidad de filas de x, excede las de y.
                            x = x.head(y.shape[0])
                            st.info(len_info_msg(1))
                        elif y.shape[0] > x.shape[0]:  # La cantidad de filas de y, excede las de x.
                            y = y.head(x.shape[0])
                            st.info(len_info_msg(2))

                        try:
                            # Test de Wilcoxon
                            wilcoxon_data = Wilcoxon(x, y)
                            wilcoxon_data.test()

                            # Mostrar Resultado
                            df_mostrar = build_df_test_result(statistic=wilcoxon_data.statistic,
                                                              p_value=wilcoxon_data.p_value)
                            st.dataframe(df_mostrar,
                                         # column_config=colm_template(["Statistics", "Valor de P"]),
                                         hide_index=True, use_container_width=True)
                        except Exception as e:
                            st.exception(e)

    with comparacion_friedman:
        st.header("Prueba de Friedman")

        file_upl_experimentos: list[UploadedFile]
        dataframes_experimentos: list[DataFrame]

        # File Uploader.
        with st.container():
            file_upl_experimentos = st.file_uploader(label="Experimentos", type=[".csv"], accept_multiple_files=True)
            dataframes_experimentos = bin_to_df(file_upl_experimentos)

        # Columna a comparar.
        opcion_col_comparacion = st.selectbox("Escoja una columna para comparar", VARIABLES_EXPERIMENTO, key=3)
        boton_comparacion = st.button("Realizar prueba de Friedman", type="primary", use_container_width=True, key=4)

        with st.container():
            if boton_comparacion:
                if len(file_upl_experimentos) == 0:
                    st.warning("No se han cargado datos de resultados de experimentos para realizar la prueba.")
                else:
                    uneven_sample_fix_tuple = fix_uneven([df[opcion_col_comparacion] for df in dataframes_experimentos])

                    samples_selection = uneven_sample_fix_tuple[0]
                    min_len = uneven_sample_fix_tuple[1]
                    if min_len != -1:
                        st.info(f"Se eliminaron filas de las tablas de los experimentos para realizar el examen. \
                        Todas las tablas pasaron a tener {min_len} filas.")

                    try:
                        # Test de Friedman.
                        friedman_result = Friedman(*samples_selection)
                        friedman_result.test()

                        # Mostrar Resultado.
                        df_mostrar = build_df_test_result(statistic=friedman_result.statistic,
                                                          p_value=friedman_result.p_value)
                        st.dataframe(df_mostrar, hide_index=True, use_container_width=True)
                    except Exception as e:
                        st.exception(e)
