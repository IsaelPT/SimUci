from datetime import datetime

from scipy.stats import wilcoxon, friedmanchisquare

from constants import *
from experiment import *
from helpers import *

#### TABS
simulacion_tab, db_tab, comparaciones_tab, validacion_tab = st.tabs(
    ("Simulación", "Datos reales", "Comparaciones", "Validación"))

with simulacion_tab:
    ############
    # Paciente #
    ############
    st.header("Paciente")

    # ID Paciente
    # IMPORTANT: EL ID DEL PACIENTE ESTÁ ALMACENADO DENTRO DEL SESSION_STATE!
    if "id_paciente" not in st.session_state:
        st.session_state.id_paciente = generate_id()
    # col1_nuevo_paciente, col2_nuevo_paciente, col3_nuevo_paciente = st.columns([1, 1, 1])
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
        opcion_tiempo_vam: int = st.number_input("Tiempo de VA", min_value=T_VAM_MIN, max_value=T_VAM_MAX,
                                                 value=T_VAM_DEFAULT)
        opcion_estad_uti: int = st.number_input("Estadía UTI", min_value=ESTAD_UTI_MIN, max_value=ESTAD_UTI_MAX,
                                                value=ESTAD_UTI_DEFAULT, help=HELP_MSG_ESTAD_UTI)
    with col2_paciente:
        opcion_insuf_resp: str = st.selectbox("Tipo Insuficiencia Respiratoria", tuple(INSUF_RESP.values()),
                                              index=1)
        opcion_tipo_vam: str = st.selectbox("Tipo de VA", tuple(TIPO_VENT.values()), )
        opcion_estad_preuti: int = st.number_input("Estadía Pre-UTI", min_value=ESTAD_PREUTI_MIN,
                                                   max_value=ESTAD_PREUTI_MAX, value=ESTAD_PREUTI_DEFAULT,
                                                   help=HELP_MSG_ESTAD_PREUTI)
        porciento = st.number_input("Porciento", min_value=PORCIENTO_SIM_MIN, max_value=PORCIENTO_SIM_MAX,
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
            st.warning(f"Todos los diagnósticos están vacíos. Mínimo incluya un diagnóstico para la simulación.")
        if not value_is_zero(insuf_resp):  # Insuficiencia Respiratoria OK?
            insuf_ok = True
        else:
            st.warning(f"Seleccione un tipo de Insuficiencia Respiratoria.")

        # Desarrollo de Simulación.
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

                st.success(f"La simulación ha concluido tras haber completado {corridas_sim} iteraciones.")
                st.rerun()
            except Exception as e:
                st.exception(f"No se pudo efectuar la simulación.\n{e}")

with db_tab:
    st.markdown("Este es el conjunto de datos que se utilizan para realizar las pruebas de comparaciones.\
            Son datos que han sido recopilados de pacientes reales en estudios anteriores realizados.")
    st.dataframe(pd.read_csv(RUTA_FICHERODEDATOS_CSV))

with comparaciones_tab:
    comparacion_wilcoxon = st.expander("Comparación vía Wilcoxon", expanded=True)
    comparacion_friedman = st.expander("Comparación vía Friedman", expanded=True)

    with comparacion_wilcoxon:
        st.header("Prueba de Wilcoxon")

        file_upl1: UploadedFile
        file_upl2: UploadedFile
        df_experimento1 = pd.DataFrame()
        df_experimento2 = pd.DataFrame()

        # Subir datos con File Uploader.
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

        # Selección de columna para comparación
        opcion_col_comparacion = st.selectbox("Escoja una columna para comparar", VARIABLES_EXPERIMENTO, key=1)
        boton_comparacion = st.button("Realizar prueba de Wilcoxon", type="primary", use_container_width=True, key=2)

        # Comparación
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
                        len_warning_msg = lambda \
                                exp: f"Se eliminaron filas del experimento {exp} para coincidir \
                                con el experimento {2 if exp == 1 else 1} ({len_dif} filas diferentes)."
                        len_dif = abs(len(x) - len(y))
                        if x.shape[0] > y.shape[0]:  # La cantidad de filas de x, excede las de y.
                            x = x.head(y.shape[0])
                            st.warning(len_warning_msg(1))
                        elif y.shape[0] > x.shape[0]:  # La cantidad de filas de y, excede las de x.
                            y = y.head(x.shape[0])
                            st.warning(len_warning_msg(2))

                        # Test de Wilcoxon
                        try:
                            wilcoxon_data = (x, y)
                            wilcoxon_result = wilcoxon(x, y)

                            # Mostrar Resultado
                            df_mostrar = build_df_test_result(wilcoxon_result[0], wilcoxon_result[1])
                            st.data_editor(df_mostrar,
                                           column_config=colm_template(["Statistics", "Valor de P"]),
                                           disabled=True, hide_index=True, use_container_width=True)
                        except Exception as e:
                            st.exception(e)

    with comparacion_friedman:
        st.header("Prueba de Friedman")

        file_upl_experimentos: list[UploadedFile]
        dataframes_experimentos: list[DataFrame]

        with st.container():
            file_upl_experimentos = st.file_uploader(label="Experimentos", type=[".csv"], accept_multiple_files=True)
            dataframes_experimentos = bin_to_df(file_upl_experimentos)

        # Selección de columna para comparación
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
                        st.warning(f"Se eliminaron filas de las tablas de los experimentos para realizar el examen, \
                        debido a que no todas tenían la misma cantidad. \
                        Todas las tablas pasaron a tener {min_len} filas.")

                    # Test de Friedman
                    try:
                        friedman_result = friedmanchisquare(*samples_selection)

                        # Mostrar Resultado
                        df_mostrar = build_df_test_result(friedman_result[0], friedman_result[1])
                        st.data_editor(df_mostrar,
                                       column_config=colm_template(["Statistics", "Valor de P"]),
                                       hide_index=True, disabled=True, use_container_width=True)
                    except Exception as e:
                        st.exception(e)

with validacion_tab:
    st.markdown("En desarrollo... ⚠️")

    #### WIP
    # data = pd.read_csv(RUTA_FICHERODEDATOS_CSV)
    #
    # seleccion_fila = st.number_input("Selecciona una fila", min_value=1, max_value=data.shape[0] - 1, value=1)
    #
    # st.caption("Datos seleccionados")
    # st.dataframe(data.iloc[[seleccion_fila]])
    #
    # r_data: tuple = get_real_data(RUTA_FICHERODEDATOS_CSV, fila_seleccion=seleccion_fila)
    #
    # if not st.session_state.df_resultado.empty:
    #     df_experimento_sim = st.session_state.df_resultado
    #     with st.container():
    #         st.dataframe(r_data, use_container_width=True, height=200)
    #         col1_wilcoxon, col2_wilcoxon = st.columns(2)
    #         with col1_wilcoxon:
    #             st.caption("Datos reales obtenidos de la Base de Datos")
    #             e = start_experiment(
    #                 corridas_sim,
    #                 edad=r_data[0],
    #                 d1=r_data[1],
    #                 d2=r_data[2],
    #                 d3=r_data[3],
    #                 d4=r_data[4],
    #                 apache=r_data[5],
    #                 insuf_resp=r_data[6],
    #                 va=r_data[7],
    #                 t_vam=r_data[8],
    #                 est_uti=r_data[9],
    #                 est_preuti=r_data[10],
    #                 porciento=porciento
    #             )
    #             df_experimento_real = e.mean()
    #             st.dataframe(df_experimento_real, use_container_width=True)
    #         with col2_wilcoxon:
    #             st.caption("Media de datos resultantes de la Simulación")
    #             df_experimento_sim = pd.DataFrame(df_experimento_sim.mean(), columns=["Valor"])
    #             df_experimento_sim.index.name = "Datos"
    #             st.dataframe(df_experimento_sim, use_container_width=True)
    # else:
    #     st.error("No se ha realizado ninguna simulación hasta el momento. Diríjase al apartado \"Simulación\".")
