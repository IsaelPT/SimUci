from datetime import datetime
import os
import pandas as pd
from pandas import DataFrame
import streamlit as st
from streamlit.runtime.uploaded_file_manager import UploadedFile

from utils.helpers import (
    generate_id,
    get_prediction_data,
    key_categ,
    predict,
    value_is_zero,
    start_experiment,
    build_df_stats,
    format_df_time,
    format_value_for_display,
    simulate_real_data,
    bin_to_df,
    adjust_df_sizes,
    build_df_test_result,
)
from utils.constants import (
    EDAD_MIN,
    EDAD_MAX,
    EDAD_DEFAULT,
    APACHE_MIN,
    APACHE_MAX,
    APACHE_DEFAULT,
    HELP_MSG_APACHE,
    HELP_MSG_TIEMPO_VAM,
    RUTA_PREDICCIONES_CSV,
    T_VAM_MIN,
    T_VAM_MAX,
    T_VAM_DEFAULT,
    ESTAD_UTI_MIN,
    ESTAD_UTI_MAX,
    ESTAD_UTI_DEFAULT,
    HELP_MSG_ESTAD_UTI,
    INSUF_RESP,
    TIPO_VENT,
    ESTAD_PREUTI_MIN,
    ESTAD_PREUTI_MAX,
    ESTAD_PREUTI_DEFAULT,
    HELP_MSG_ESTAD_PREUTI,
    PORCIENTO_SIM_MIN,
    PORCIENTO_SIM_MAX,
    PORCIENTO_SIM_DEFAULT,
    HELP_MSG_PORCIENTO_SIM,
    DIAG_PREUCI,
    RUTA_FICHERODEDATOS_CSV,
    PRIMARY_COLOR,
    VARIABLES_EXPERIMENTO,
    CORRIDAS_SIM_MIN,
    CORRIDAS_SIM_MAX,
    CORRIDAS_SIM_DEFAULT,
    HELP_MSG_CORRIDA_SIM,
    INFO_STATISTIC,
    INFO_P_VALUE,
)
from uci.stats import Wilcoxon, Friedman, StatsUtils
import traceback


########
# TABS #
########
simulacion_tab, datos_reales_tab, comparaciones_tab = st.tabs(
    ("Simulación", "Datos reales", "Comparaciones")
)

with simulacion_tab:
    ############
    # Paciente #
    ############
    st.header("Paciente")

    # ID Paciente
    # NOTE: EL ID DEL PACIENTE ESTÁ ALMACENADO DENTRO DEL SESSION_STATE!
    if "id_paciente" not in st.session_state:
        st.session_state.id_paciente = generate_id()
    if "semilla_simulacion" not in st.session_state:
        st.session_state.semilla_simulacion = 0

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
    col1_paciente, col2_paciente = st.columns(2)
    with col1_paciente:
        col1a_paciente, col1b_paciente = st.columns(2)
        with col1a_paciente:
            opcion_edad: int = st.number_input(
                label="Edad", min_value=EDAD_MIN, max_value=EDAD_MAX, value=EDAD_DEFAULT
            )
            opcion_tiempo_vam: int = st.number_input(
                label="Tiempo VA",
                min_value=T_VAM_MIN,
                max_value=T_VAM_MAX,
                value=T_VAM_DEFAULT,
                help=HELP_MSG_TIEMPO_VAM,
            )
            opcion_estad_preuti: int = st.number_input(
                label="Tiempo Pre-UCI",
                min_value=ESTAD_PREUTI_MIN,
                max_value=ESTAD_PREUTI_MAX,
                value=ESTAD_PREUTI_DEFAULT,
                help=HELP_MSG_ESTAD_PREUTI,
            )
        with col1b_paciente:
            opcion_apache: int = st.number_input(
                label="Apache",
                min_value=APACHE_MIN,
                max_value=APACHE_MAX,
                value=APACHE_DEFAULT,
                help=HELP_MSG_APACHE,
            )
            opcion_estad_uti: int = st.number_input(
                label="Tiempo UCI",
                min_value=ESTAD_UTI_MIN,
                max_value=ESTAD_UTI_MAX,
                value=ESTAD_UTI_DEFAULT,
                help=HELP_MSG_ESTAD_UTI,
            )
            input_porciento = st.number_input(
                label="Porciento Tiempo UCI",
                min_value=PORCIENTO_SIM_MIN,
                max_value=PORCIENTO_SIM_MAX,
                value=PORCIENTO_SIM_DEFAULT,
                help=HELP_MSG_PORCIENTO_SIM,
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
        col1_diag, col2_diagn = st.columns(2)
        with col1_diag:
            with st.popover("Diagnósticos Ingreso"):
                st.markdown("Seleccione los Diagnósticos de Ingreso del paciente:")
                opcion_diag_ing1: str = st.selectbox(
                    label="Diagnóstico 1",
                    options=tuple(DIAG_PREUCI.values()),
                    index=0,
                    key="diag-ing-1",
                )
                opcion_diag_ing2: str = st.selectbox(
                    label="Diagnóstico 2",
                    options=tuple(DIAG_PREUCI.values()),
                    index=0,
                    key="diag-ing-2",
                )
                opcion_diag_ing3: str = st.selectbox(
                    label="Diagnóstico 3",
                    options=tuple(DIAG_PREUCI.values()),
                    index=0,
                    key="diag-ing-3",
                )
                opcion_diag_ing4: str = st.selectbox(
                    label="Diagnóstico 4",
                    options=tuple(DIAG_PREUCI.values()),
                    index=0,
                    key="diag-ing-4",
                )
        with col2_diagn:
            with st.popover("Diagnósticos Egreso"):
                st.markdown("Seleccione los Diagnósticos de Egreso del paciente:")
                # opcion_diag_egreso1: str = st.selectbox(
                #     label="Diagnóstico 1",
                #     options=tuple(DIAG_PREUCI.values()),
                #     index=0,
                #     key="diag-egreso-1",
                # )
                opcion_diag_egreso2: str = st.selectbox(
                    label="Diagnóstico 2",
                    options=tuple(DIAG_PREUCI.values()),
                    index=0,
                    key="diag-egreso-2",
                )
                # opcion_diag_egreso3: str = st.selectbox(
                #     label="Diagnóstico 3",
                #     options=tuple(DIAG_PREUCI.values()),
                #     index=0,
                #     key="diag-egreso-3",
                # )
                # opcion_diag_egreso4: str = st.selectbox(
                #     label="Diagnóstico 4",
                #     options=tuple(DIAG_PREUCI.values()),
                #     index=0,
                #     key="diag-egreso-4",
                # )

        # Datos Paciente Recolectados (Son los datos de entrada para ser procesados).
        edad: int = opcion_edad
        apache: int = opcion_apache
        diag_ing1: int = key_categ("diag", opcion_diag_ing1)
        diag_ing2: int = key_categ("diag", opcion_diag_ing2)
        diag_ing3: int = key_categ("diag", opcion_diag_ing3)
        diag_ing4: int = key_categ("diag", opcion_diag_ing4)
        # diag_egreso1: int = key_categ("diag", opcion_diag_egreso1)
        diag_egreso2: int = key_categ("diag", opcion_diag_egreso2)
        # diag_egreso3: int = key_categ("diag", opcion_diag_egreso3)
        # diag_egreso4: int = key_categ("diag", opcion_diag_egreso4)
        tipo_vam: int = key_categ("va", opcion_tipo_vam)
        tiempo_vam: int = opcion_tiempo_vam
        estadia_uti: int = opcion_estad_uti
        estadia_preuti: int = opcion_estad_preuti
        insuf_resp: int = key_categ("insuf", opcion_insuf_resp)

    st.divider()

    ##############
    # Simulación #
    ##############
    st.header("Simulación")

    diag_ok = False
    insuf_ok = False
    resultado_experimento = pd.DataFrame()

    # session_state para visualizar datos simulación.
    if "df_resultado" not in st.session_state:
        st.session_state.df_resultado = pd.DataFrame()

    with st.container():
        corridas_sim = st.number_input(
            "Corridas de la Simulación",
            min_value=CORRIDAS_SIM_MIN,
            max_value=CORRIDAS_SIM_MAX,
            value=CORRIDAS_SIM_DEFAULT,
            help=HELP_MSG_CORRIDA_SIM,
        )
        boton_comenzar_simulacion = st.button(
            "Comenzar Simulación", type="primary", use_container_width=True
        )

    # Mostrar DataFrame con resultado de la simulación para ese paciente.
    if not st.session_state.df_resultado.empty:
        toggle_format = st.toggle(
            "Tiempo en días",
            value=True,
            help="Formato de Tiempo. Al activar esta opción se muestran los días convertidos en horas.",
            key="formato-tiempo-simulacion",
        )
        df_simulacion = build_df_stats(
            data=st.session_state.df_resultado,
            sample_size=corridas_sim,
            include_mean=True,
            include_std=True,
            include_confint=True,
            include_metrics=True,
            include_info_label=True,
        )

        # Aplicar formato si está habilitado
        if toggle_format:
            # Crear una copia del DataFrame para mostrar
            display_df = df_simulacion.copy()

            # Aplicar formato a todas las columnas que contengan 'Tiempo' en el nombre para todas las filas
            for col in display_df.columns:
                if "Tiempo" in col and display_df[col].dtype != object:
                    display_df[col] = display_df[col].apply(format_value_for_display)
        else:
            display_df = df_simulacion

        st.dataframe(
            display_df,
            hide_index=True,
            use_container_width=True,
        )

        # Mostrar predicción de clases y porcentaje.
        if (
            "prediccion_clases" in st.session_state
            and "prediccion_porcentaje" in st.session_state
        ):
            col1, col2 = st.columns(2)
            with col1:
                # CLASES PREVIEW
                st.markdown(
                    "### Paciente no fallece"
                    if st.session_state.prediccion_clases == 0
                    else "### Paciente fallece"
                )
            with col2:
                # METRIC PREVIEW
                prev_pred = st.session_state.get("prev_prediccion_porcentaje", None)
                current_pred = st.session_state.prediccion_porcentaje
                delta_label = ""
                delta_color = "normal"
                if prev_pred is not None:
                    diff = current_pred - prev_pred
                    if diff > 0:
                        delta_label = f"Aumento de {round(abs(diff) * 100)}%"
                        delta_color = "normal"
                    elif diff < 0:
                        delta_label = f"Disminución de {round(abs(diff) * 100)}%"
                        delta_color = "inverse"
                    else:
                        delta_label = "Sin cambio"
                        delta_color = "off"
                else:
                    delta_label = None
                    delta_color = "normal"

                st.metric(
                    label="Probabilidad",
                    value=f"{round(current_pred * 100)}%",
                    delta=delta_label,
                    delta_color=delta_color,
                    border=True,
                )

                st.session_state.prev_prediccion_porcentaje = current_pred

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

    if boton_comenzar_simulacion:
        # Validación de campos para realizar simulación.
        if not value_is_zero(
            [diag_ing1, diag_ing2, diag_ing3, diag_ing4]
        ):  # campos de Diagnósticos OK?
            diag_ok = True
        else:
            st.warning(
                "Todos los diagnósticos están vacíos. Se debe incluir mínimo un diagnóstico para realizar la simulación."
            )

        if not value_is_zero(insuf_resp):  # campo de dInsuficiencia Respiratoria OK?
            insuf_ok = True
        else:
            st.warning("Seleccione un tipo de Insuficiencia Respiratoria.")

        # Desarrollo de la SIMULACIÓN.
        if diag_ok and insuf_ok:
            try:
                # Experimento / Simulación.
                resultado_experimento = start_experiment(
                    corridas_sim,
                    edad,
                    diag_ing1,
                    diag_ing2,
                    diag_ing3,
                    diag_ing4,
                    apache,
                    insuf_resp,
                    insuf_resp,
                    estadia_uti,
                    tiempo_vam,
                    estadia_preuti,
                    input_porciento,
                )

                # Predicción de clases y porcentaje.
                try:
                    if "prediccion_clases" not in st.session_state:
                        st.session_state.prediccion_clases = 0
                    if "prediccion_porcentaje" not in st.session_state:
                        st.session_state.prediccion_porcentaje = 0.0

                    df_to_predict = get_prediction_data(
                        {
                            "Edad": edad,
                            "Diag.Ing1": diag_ing1,
                            "Diag.Ing2": diag_ing2,
                            "Diag.Egr2": diag_egreso2,
                            "TiempoVAM": tiempo_vam,
                            "APACHE": apache,
                        }
                    )
                    prediction = predict(df_to_predict)

                    if prediction is not None:
                        st.session_state.prediccion_clases = prediction[0][0]
                        st.session_state.prediccion_porcentaje = prediction[1][0]
                    else:
                        raise ValueError(
                            "No se pudo realizar la predicción de clases y porcentaje."
                        )

                    print(
                        f"Predicción: {st.session_state.prediccion_clases}, Porcentaje: {st.session_state.prediccion_porcentaje}"
                    )
                except Exception as e:
                    st.error(
                        f"No se pudo realizar la predicción de clases y porcentaje. Error asociado: {e}"
                    )

                # Guardar resultados (forma local del proyecto).
                path_base = f"experiments\\paciente-id-{st.session_state.id_paciente}"
                if not os.path.exists(path_base):
                    os.makedirs(path_base)
                fecha: str = datetime.now().strftime("%d-%m-%Y")
                path: str = (
                    f"{path_base}\\experimento-id {generate_id(5)} fecha {fecha} corridas {corridas_sim}.csv"
                )
                resultado_experimento.to_csv(path, index=False)
                st.session_state.df_resultado = resultado_experimento

                st.rerun()
            except Exception as data:
                st.exception(
                    f"No se pudo efectuar la simulación. Error asociado: \n{data}"
                )

###############################
# SIMULACION CON DATOS REALES #
###############################
with datos_reales_tab:
    ##############
    # Validación #
    ##############
    st.header("Validación con Datos Reales")

    # fijar_semilla_toggle = st.toggle(
    #     "Fijar semilla",
    #     value=False,
    #     help="Al fijar una semilla, se usará una sola semilla para las simulaciones. Los resultados se vuelve reproducibles."
    # )

    # if fijar_semilla_toggle:
    #     fix_seed(1)

    df_data = pd.read_csv(RUTA_FICHERODEDATOS_CSV)
    df_data.index.name = "Paciente"

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

    # El `df_selection` devuelve un dict de la forma:
    # {
    #     "selection": {
    #         "rows": [0, 1, 2, ...],
    #         "columns": [0, 1, 2, ...]
    #     }
    # }

    # Si se seleccionó alguna fila, se asigna a esta variable. De no seleccionarse nada es None.
    selection = df_selection[0] if df_selection else None

    if "df_sim_datos_reales" not in st.session_state:
        st.session_state.df_sim_datos_reales = pd.DataFrame()

    # DataFrame se encuentra con resultado de experimento con datos reales
    if selection == 0 or selection is not None:
        st.markdown("Resultados de simulación con paciente seleccionado")

        corridas_sim = CORRIDAS_SIM_DEFAULT

        col1, col2 = st.columns(2)
        with col1:
            toggle_format = st.toggle(
                "Tiempo en días",
                value=False,
                help="Formato de tiempo. Al activar muestra las horas en su aproximación a lo que sería en días.",
            )
        with col2:
            with st.popover(
                label="Cantidad de Simulaciones",
                use_container_width=True,
                help=HELP_MSG_CORRIDA_SIM,
            ):
                corridas_sim = st.number_input(
                    label="Cantidad de Simulaciones",
                    min_value=CORRIDAS_SIM_MIN,
                    max_value=CORRIDAS_SIM_MAX,
                    value=CORRIDAS_SIM_DEFAULT,
                    help=HELP_MSG_CORRIDA_SIM,
                )

        data: tuple[float] = simulate_real_data(
            ruta_fichero_csv=RUTA_FICHERODEDATOS_CSV, df_selection=selection
        )

        df_sim_datos_reales = build_df_stats(
            data,
            corridas_sim,
            include_mean=True,
            include_std=True,
            include_confint=True,
            include_metrics=True,
            include_info_label=True,
        )

        if toggle_format:
            st.dataframe(
                format_df_time(df_sim_datos_reales),
                hide_index=True,
                use_container_width=True,
            )
        else:
            st.dataframe(
                df_sim_datos_reales,
                hide_index=True,
                use_container_width=True,
            )

    # Simular todos los datos en la tabla.
    if st.button(
        "Simular todos los datos en la tabla",
        type="primary",
        use_container_width=True,
        key="simular_tabla",
    ):
        with st.spinner(
            text="Simulando todos los datos en la tabla. Esto puede tardar varios minutos...",
            show_time=True,
        ):
            try:
                lista_experimentos: list[tuple[float]] = simulate_real_data(
                    RUTA_FICHERODEDATOS_CSV, df_selection=-1
                )

                # DataFrame con todos los resultados de simulaciones a todos los pacientes en la tabla.
                df_sim_datos_reales = build_df_stats(
                    lista_experimentos,
                    CORRIDAS_SIM_DEFAULT,
                    include_mean=True,
                    include_std=False,
                    include_confint=False,
                    include_info_label=False,
                )
                df_sim_datos_reales.index.name = "Paciente"
                st.session_state.df_sim_datos_reales = df_sim_datos_reales

                st.toast(
                    "Se ha realizado la simulación a cada paciente de la tabla.",
                    icon="✅",
                )
            except Exception as e:
                st.toast(
                    f"Se ha producido un error al realizar la simulación: {e}",
                    icon="⚠️",
                )

    # Mostrar simulación con datos reales.
    if not st.session_state.df_sim_datos_reales.empty:
        # Usar el mismo formato que en la sección de simulación
        toggle_format_reales = st.toggle(
            "Tiempo en días",
            value=True,
            help="Formato de Tiempo. Al activar esta opción se muestran los días convertidos en horas.",
            key="formato-tiempo-reales",
        )

        # Crear una copia para mostrar sin modificar los datos originales
        display_df_reales = st.session_state.df_sim_datos_reales.copy()

        # Aplicar formato si está habilitado
        if toggle_format_reales:
            # Aplicar formato a todas las columnas que contengan 'Tiempo' en el nombre
            for col in display_df_reales.columns:
                if "Tiempo" in col:
                    display_df_reales[col] = display_df_reales[col].apply(
                        format_value_for_display
                    )

        st.dataframe(
            display_df_reales,
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
            file_name="Experimentos con Datos Reales.csv",
            mime="text/csv",
            use_container_width=True,
            key="guardar_sim_datos_reales",
        )

    st.divider()
    st.markdown("# Validación de la predicción para todos los Datos Reales")

    if st.button(label="Validar predicción", type="primary", use_container_width=True):
        with st.spinner(
            text="Validando predicción para valores reales. Esto puede tardar unos minutos...",
            show_time=True,
        ):
            try:
                df_data_pred = pd.read_csv(RUTA_PREDICCIONES_CSV)

                df_predic = get_prediction_data(df_data_pred)
                df_predic = predict(df_predic)

                with st.expander("Datos de entrada"):
                    st.dataframe(
                        df_data_pred,
                        hide_index=True,
                        use_container_width=True,
                    )

                # Asegurar que y_true sea un array numérico sin nulos
                y_true = (
                    pd.to_numeric(df_data_pred["Fallece"], errors="coerce")
                    .fillna(-1)
                    .to_numpy()
                )
                # Probabilidades de la clase positiva
                y_pred_proba = df_predic[1]

                intervals = [0.8, 0.9, 0.95]

                calibration_metric = StatsUtils.calibration_metric_predict(
                    y_true, y_pred_proba, intervals
                )

                st.dataframe(
                    pd.DataFrame(
                        {
                            "Intervalo": intervals,
                            "Muestras dentro del intervalo": calibration_metric,
                        }
                    ),
                    hide_index=True,
                    use_container_width=True,
                )
            except Exception as e:
                st.error(f"Error al validar la predicción: {e}")
                st.exception(traceback.format_exc())

#################
# COMPARACIONES #
#################
with comparaciones_tab:
    wilcoxon_tab, friedman_tab = st.tabs(("Wilcoxon", "Friedman"))

    with wilcoxon_tab:
        st.markdown("### Wilcoxon")

        file_upl1: UploadedFile
        file_upl2: UploadedFile
        df_experimento1 = pd.DataFrame()
        df_experimento2 = pd.DataFrame()

        with st.container():
            col1, col2 = st.columns(2)
            with col1:
                file_upl1 = st.file_uploader(
                    label="Experimento 1", type=[".csv"], accept_multiple_files=False
                )
            with col2:
                file_upl2 = st.file_uploader(
                    label="Experimento 2", type=[".csv"], accept_multiple_files=False
                )

            with st.expander("Previsualización", expanded=False):
                if file_upl1:
                    df_experimento1 = bin_to_df(file_upl1)
                    if not df_experimento1.empty:
                        st.dataframe(
                            df_experimento1,
                            height=200,
                            use_container_width=True,
                            hide_index=True,
                        )
                if file_upl2:
                    df_experimento2 = bin_to_df(file_upl2)
                    if not df_experimento2.empty:
                        st.dataframe(
                            df_experimento2,
                            height=200,
                            use_container_width=True,
                            hide_index=True,
                        )

        # Columna a comparar.
        opcion_col_comparacion = st.selectbox(
            "Seleccione una columna para comparar",
            VARIABLES_EXPERIMENTO,
            key="col-comparacion-wilcoxon",
        )
        boton_comparacion = st.button(
            "Realizar prueba de Wilcoxon",
            type="primary",
            use_container_width=True,
            key="boton-comparacion-wilcoxon",
        )

        # Comparación Wilcoxon.
        with st.container():
            if boton_comparacion:
                if df_experimento1.empty or df_experimento2.empty:
                    st.warning(
                        "No se puede realizar la comparación. Se detectan datos vacíos o falta de datos en los experimentos."
                    )
                else:
                    x: DataFrame = df_experimento1[opcion_col_comparacion]
                    y: DataFrame = df_experimento2[opcion_col_comparacion]
                    if x.equals(y):
                        st.error(
                            'Imposible realizar prueba de Wilcoxon cuando la diferencia entre los elementos de "x" y "y" es cero para todos los elementos. Verifique que no cargó el mismo experimento dos veces.'
                        )
                    else:
                        # Corrección de que existen la misma cantidad de filas en ambas tablas.
                        len_dif = abs(len(x) - len(y))

                        # Ajustar el tamaño de las muestras si es necesario
                        if x.shape[0] > y.shape[0]:  # X mayor que Y
                            x = x.head(y.shape[0])
                            st.info(
                                f"Se eliminaron filas del experimento 1 para coincidir con el experimento 2 ({len_dif} filas diferentes)."
                            )
                        elif x.shape[0] < y.shape[0]:  # Y mayor que X
                            y = y.head(x.shape[0])
                            st.info(
                                f"Se eliminaron filas del experimento 2 para coincidir con el experimento 1 ({len_dif} filas diferentes)."
                            )

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
                            st.markdown(INFO_STATISTIC)
                            st.markdown(INFO_P_VALUE)
                        except Exception as data:
                            st.exception(data)
    with friedman_tab:
        st.markdown("### Friedman")

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
            "Seleccione una columna para comparar", VARIABLES_EXPERIMENTO, key=3
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
                        "No se han cargado datos de resultados de experimentos para realizar esta prueba."
                    )
                elif not len(file_upl_experimentos) >= 3:
                    st.warning(
                        "Debe cargar más de 3 muestras para realizar esta prueba."
                    )
                else:
                    adjusted_sample_tuple = adjust_df_sizes(
                        [df[opcion_col_comparacion] for df in dataframes_experimentos]
                    )

                    samples_selection = adjusted_sample_tuple[0]
                    min_sample_size = adjusted_sample_tuple[1]
                    if min_sample_size != -1:
                        st.info(
                            f"Para realizar correctamente el examen se eliminaron filas de las tablas de los experimentos, ya que es un requisito que exista el mismo tamaño de muestra. Todas las tablas pasaron a tener {min_sample_size} filas."
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
                        st.markdown(INFO_STATISTIC)
                        st.markdown(INFO_P_VALUE)
                    except Exception as data:
                        st.exception(data)
