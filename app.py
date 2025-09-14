from datetime import datetime
import os
import pandas as pd
from pandas import DataFrame
import streamlit as st
from streamlit.runtime.uploaded_file_manager import UploadedFile

from utils.helpers import (
    adjust_df_sizes,
    bin_to_df,
    build_df_for_stats,
    build_df_test_result,
    fix_seed,
    format_time_columns,
    generate_id,
    get_data_for_prediction,
    key_categ,
    predict,
    simulate_real_data,
    start_experiment,
    value_is_zero,
    prepare_patient_data_for_prediction,
    extract_real_data,
    apply_theme,
)
from utils.constants import (
    EDAD_MIN,
    EDAD_MAX,
    EDAD_DEFAULT,
    APACHE_MIN,
    APACHE_MAX,
    APACHE_DEFAULT,
    HELP_MSG_APACHE,
    HELP_MSG_PREDICCION_METRIC,
    HELP_MSG_TIEMPO_VAM,
    HELP_MSG_TIME_FORMAT,
    LABEL_PREDICTION_METRIC,
    LABEL_TIME_FORMAT,
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
    EXPERIMENT_VARIABLES as EXP_VARIABLES,
    CORRIDAS_SIM_MIN,
    CORRIDAS_SIM_MAX,
    CORRIDAS_SIM_DEFAULT,
    HELP_MSG_CORRIDA_SIM,
    INFO_STATISTIC,
    INFO_P_VALUE,
)
from uci.stats import Wilcoxon, Friedman


# Configuración inicial de la página
st.set_page_config(page_title="SimUci", page_icon="🏥", layout="wide", initial_sidebar_state="expanded")

if "theme" not in st.session_state:
    st.session_state.theme = "light"

apply_theme(st.session_state.theme)

with st.sidebar:
    #
    # TITLE
    #
    st.header(body="SimUci", anchor=False, width="stretch", divider="gray")

    #
    # APP INFORMATION
    #
    st.subheader("Sobre la App")
    st.markdown("""
    Esta aplicación permite:
    - Simular pacientes UCI
    - Analizar datos reales
    - Realizar predicciones
    - Comparar resultados estadísticos
    """)

    st.divider()

    #
    # GLOBAL SEED CONFIG
    #
    st.subheader("Semilla Global de Simulación")
    with st.expander(label="Explicación", expanded=False):
        st.caption(
            body="Valor global de la semilla aleatoria utilizada para realizar las simulaciones. Esta determina que los resultados de las simulaciones resulten predecibles o no.",
            width="stretch",
        )
    if "global_sim_seed" not in st.session_state:
        st.session_state.global_sim_seed = 0
    st.session_state.global_sim_seed = st.number_input(
        label="Semilla",
        value=st.session_state.global_sim_seed,
        min_value=0,
        max_value=999_999,
    )
    if st.button(
        icon="🔄️",
        label="Restablecer",
        disabled=True if st.session_state.global_sim_seed == 0 else False,
        type="secondary",
        use_container_width=True,
    ):
        st.session_state.global_sim_seed = 0

    st.divider()

    #
    # THEME CONFIG
    #
    st.subheader("Tema")
    theme_toggle = st.toggle(
        label="Modo Oscuro",
    )
    new_theme = "dark" if theme_toggle else "light"
    if new_theme != st.session_state.theme:
        st.session_state.theme = new_theme
        apply_theme(new_theme)
        st.rerun()

    st.divider()

    #
    # VERSION INFO
    #
    st.header("Versión")
    st.caption("Beta 0.2 - Septiembre 2025")

########
# TABS #
########
simulacion_tab, datos_reales_tab, comparaciones_tab = st.tabs(("Simulación", "Datos reales", "Comparaciones"))

# Inicializar session_state para datos reales
if "df_sim_datos_reales" not in st.session_state:
    st.session_state.df_sim_datos_reales = pd.DataFrame()
if "df_pacientes_individuales" not in st.session_state:
    st.session_state.df_pacientes_individuales = pd.DataFrame()
if "friedman_results" not in st.session_state:
    st.session_state.friedman_results = {}

with simulacion_tab:
    ############
    # Paciente #
    ############

    # ID Paciente
    # NOTE: EL ID DEL PACIENTE ESTÁ ALMACENADO DENTRO DEL SESSION_STATE!
    if "id_paciente" not in st.session_state:
        st.session_state.id_paciente = generate_id()
    if "semilla_simulacion" not in st.session_state:
        st.session_state.semilla_simulacion = 0

    pac_col1, pac_col2 = st.columns(spec=2, gap="small", border=False, vertical_alignment="bottom")
    with pac_col1:
        st.header("Paciente")
    with pac_col2:
        col1, col2 = st.columns(spec=[1, 2], gap="small", border=False)
        with col1:
            nuevo_paciente = st.button("Nuevo paciente", use_container_width=True)
            if nuevo_paciente:
                st.session_state.id_paciente = generate_id()
        with col2:
            st.session_state.id_paciente = st.text_input(
                label="ID Paciente",
                value=st.session_state.id_paciente,
                max_chars=10,
                placeholder="ID Paciente",
                label_visibility="collapsed",
            )

    col1_paciente, col2_paciente = st.columns(spec=2, border=False, gap="small")
    with col1_paciente:
        col1a_paciente, col1b_paciente, col1c_paciente = st.columns(spec=3, gap="small", border=False)
        with col1a_paciente:
            opcion_edad: int = st.number_input(label="Edad", min_value=EDAD_MIN, max_value=EDAD_MAX, value=EDAD_DEFAULT)
            opcion_estad_preuti: int = st.number_input(
                label="Tiempo Pre-UCI",
                min_value=ESTAD_PREUTI_MIN,
                max_value=ESTAD_PREUTI_MAX,
                value=ESTAD_PREUTI_DEFAULT,
                help=HELP_MSG_ESTAD_PREUTI,
            )
        with col1b_paciente:
            opcion_tiempo_vam: int = st.number_input(
                label="Tiempo VA",
                min_value=T_VAM_MIN,
                max_value=T_VAM_MAX,
                value=T_VAM_DEFAULT,
                help=HELP_MSG_TIEMPO_VAM,
            )
            input_porciento = st.number_input(
                label="Porciento Tiempo UCI",
                min_value=PORCIENTO_SIM_MIN,
                max_value=PORCIENTO_SIM_MAX,
                value=PORCIENTO_SIM_DEFAULT,
                help=HELP_MSG_PORCIENTO_SIM,
            )
        with col1c_paciente:
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
        col1x_paciente, col1y_paciente = st.columns(spec=2, gap="small", border=False)
        with col1x_paciente:
            opcion_insuf_resp: str = st.selectbox(
                label="Tipo de Insuficiencia Respiratoria",
                options=tuple(INSUF_RESP.values()),
                index=1,
            )
        with col1y_paciente:
            opcion_tipo_vam: str = st.selectbox(
                label="Tipo de Ventilación Artificial",
                options=tuple(TIPO_VENT.values()),
            )
    with col2_paciente:
        subcol1, subcol2 = st.columns(2)
        with subcol1:
            opcion_diag_ing1: str = st.selectbox(
                label="Diagnóstico Ing. 1",
                options=tuple(DIAG_PREUCI.values()),
                index=0,
                key="diag-ing-1",
            )
            opcion_diag_ing3: str = st.selectbox(
                label="Diagnóstico Ing. 3",
                options=tuple(DIAG_PREUCI.values()),
                index=0,
                key="diag-ing-3",
            )
        with subcol2:
            opcion_diag_ing2: str = st.selectbox(
                label="Diagnóstico Ing. 2",
                options=tuple(DIAG_PREUCI.values()),
                index=0,
                key="diag-ing-2",
            )
            opcion_diag_ing4: str = st.selectbox(
                label="Diagnóstico Ing. 4",
                options=tuple(DIAG_PREUCI.values()),
                index=0,
                key="diag-ing-4",
            )
        # opcion_diag_egreso1: str = st.selectbox(
        #     label="Diagnóstico 1",
        #     options=tuple(DIAG_PREUCI.values()),
        #     index=0,
        #     key="diag-egreso-1",
        # )
        opcion_diag_egreso2: str = st.selectbox(
            label="Diagnóstico Egreso 2",
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
            step=50,
            value=CORRIDAS_SIM_DEFAULT,
            help=HELP_MSG_CORRIDA_SIM,
        )
        boton_comenzar_simulacion = st.button("Comenzar Simulación", type="primary", use_container_width=True)

    # Mostrar DataFrame con resultado de la simulación para ese paciente.
    if not st.session_state.df_resultado.empty:
        toggle_format = st.toggle(
            label=LABEL_TIME_FORMAT,
            value=False,
            help=HELP_MSG_TIME_FORMAT,
            key="formato-tiempo-simulacion",
        )
        df_simulacion = build_df_for_stats(
            data=st.session_state.df_resultado,
            sample_size=corridas_sim,
            include_mean=True,
            include_std=True,
            include_confint=True,
            include_metrics=False,
            include_info_label=True,
            labels_structure={
                0: "Promedio",
                1: "Desviación Estándar",
                2: "Límite Inf",
                3: "Límite Sup",
            },
        )

        # Aplicar formato si está habilitado
        if toggle_format:
            display_df = format_time_columns(df_simulacion, exclude_rows=["Métrica de Calibración"])
        else:
            display_df = df_simulacion

        st.dataframe(
            display_df,
            hide_index=True,
            use_container_width=True,
        )

        # Mostrar predicción de clases y porcentaje.
        if "prediccion_clases" in st.session_state and "prediccion_porcentaje" in st.session_state:
            # METRIC PREVIEW
            prev_pred = st.session_state.get("prev_prediccion_porcentaje", None)

            # Asegurar que el valor actual sea un float
            current_pred = float(st.session_state.prediccion_porcentaje)

            delta_label = ""
            delta_color = "normal"

            # Lógica del valor anterior
            if prev_pred is not None:
                try:
                    prev_val = float(prev_pred)

                    # Calcular el cambio porcentual
                    change = current_pred - prev_val
                    percent_change = change * 100

                    how_chaged: str

                    if change > 0:
                        how_chaged = "Incremento"
                        delta_color = "inverse"  # Flecha verde hacia arriba
                    if change < 0:
                        how_chaged = "Disminución"
                        delta_color = "normal"  # Flecha roja hacia abajo

                    if change == 0:
                        delta_label = "Sin cambio"
                        delta_color = "off"  # Sin flecha
                    else:
                        delta_label = f"{how_chaged} de un {abs(percent_change):.0f}% de la probabilidad de fallecer del paciente respecto a la predicción anterior"

                except Exception:
                    # Si ocurre un error, no mostrar delta
                    delta_label = None
                    delta_color = "normal"

            else:
                delta_color = "normal"

            # Variable clasificación binaria (0, 1) <-> (False, True)
            paciente_vive: bool = True if st.session_state.prediccion_clases == 0 else False
            metric_display_value: str = f"{'Paciente no fallece' if paciente_vive else 'Paciente fallece'} (predicción de {(current_pred * 100):.0f}% )"

            st.metric(
                label=LABEL_PREDICTION_METRIC,
                value=metric_display_value,
                delta=delta_label,
                delta_color=delta_color,
                width="stretch",
                border=True,
                help=HELP_MSG_PREDICCION_METRIC,
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

        st.success(f"Concluyó la **simulación** tras {corridas_sim} corridas.")

    if boton_comenzar_simulacion:
        # Validación de campos para realizar simulación.
        if not value_is_zero([diag_ing1, diag_ing2, diag_ing3, diag_ing4]):  # campos de Diagnósticos OK?
            diag_ok = True
        else:
            st.warning(
                "Todos los diagnósticos están vacíos. Se debe incluir mínimo un diagnóstico para realizar la simulación."
            )

        if not value_is_zero(insuf_resp):  # campo de Insuficiencia Respiratoria OK?
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

                    df_to_predict = get_data_for_prediction(
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
                        raise ValueError("No se pudo realizar la predicción de clases y porcentaje.")

                    print(
                        f"Predicción: {st.session_state.prediccion_clases}, Porcentaje: {st.session_state.prediccion_porcentaje}"
                    )
                except Exception as e:
                    st.error(f"No se pudo realizar la predicción de clases y porcentaje. Error asociado: {e}")

                # Guardar resultados (forma local del proyecto).
                path_base = f"experiments\\paciente-id-{st.session_state.id_paciente}"
                if not os.path.exists(path_base):
                    os.makedirs(path_base)
                fecha: str = datetime.now().strftime("%d-%m-%Y")
                path: str = f"{path_base}\\experimento-id {generate_id(5)} fecha {fecha} corridas {corridas_sim}.csv"
                resultado_experimento.to_csv(path, index=False)
                st.session_state.df_resultado = resultado_experimento

                st.rerun()
            except Exception as data:
                st.exception(f"No se pudo efectuar la simulación. Error asociado: \n{data}")

###############################
# SIMULACION CON DATOS REALES #
###############################
with datos_reales_tab:
    ##############
    # Validación #
    ##############
    st.header("Validación con Datos Reales")

    html_text = f'<p style="color:{PRIMARY_COLOR};">Puede seleccionar una fila para realizar una simulación al paciente seleccionado.</p>'
    st.markdown(html_text, unsafe_allow_html=True)

    df_data = pd.read_csv(RUTA_FICHERODEDATOS_CSV)
    df_data.index.name = "Paciente"

    df_selection = st.dataframe(
        df_data,
        key="data",
        on_select="rerun",
        selection_mode=["single-row"],
        hide_index=False,
        height=300,
    )["selection"]["rows"]

    # INFORMACIÓN SOBRE SELECCIÓN
    # ------------------------------------------
    #
    # df_selection devuelve un dict de la forma:
    #
    # "df_selection": {
    #     "rows": [0, 1, 2, ...],
    #     "columns": [0, 1, 2, ...]
    # }
    #

    with st.popover(label="Configuraciones de Simulación", use_container_width=True):
        col1sim_config, col2sim_config, col3sim_config = st.columns(
            spec=[2, 1, 1], gap="small", vertical_alignment="center", border=False, width="stretch"
        )

        with col1sim_config:
            corridas_sim = st.number_input(
                label="Cantidad de Simulaciones por paciente",
                min_value=CORRIDAS_SIM_MIN,
                max_value=CORRIDAS_SIM_MAX,
                step=50,
                value=CORRIDAS_SIM_DEFAULT,
                help=HELP_MSG_CORRIDA_SIM,
            )
        with col2sim_config:
            fijar_semilla_toggle = st.toggle(
                label="Fijar semilla de simulación",
                value=False,
                help="Al fijar una semilla los resultados se vuelve reproducibles.",
            )
        with col3sim_config:
            if fijar_semilla_toggle:
                st.info(f"Semilla fijada con valor: **`{st.session_state.global_sim_seed}`**")
            else:
                st.info("Semilla desfijada")

    if fijar_semilla_toggle:
        fix_seed(st.session_state.global_sim_seed)

    corridas_sim = CORRIDAS_SIM_DEFAULT

    rerun_sim_btn = st.button(label="Correr de nuevo la simulación", type="primary", use_container_width=True)

    if "df_sim_datos_reales" not in st.session_state:
        st.session_state.df_sim_datos_reales = pd.DataFrame()
    if "prev_selection" not in st.session_state:
        st.session_state.prev_selection = None
    if "patient_data" not in st.session_state:
        st.session_state.patient_data = None

    current_selection = df_selection[0] if df_selection else None

    # Simulation and Prediction of selection
    if current_selection is not None or current_selection == 0:
        # print(f"prev >> {st.session_state.prev_selection}")
        # print(f"curr >> {current_selection}")

        if (st.session_state.prev_selection != current_selection) and rerun_sim_btn:
            # Simulation
            data = simulate_real_data(ruta_fichero_csv=RUTA_FICHERODEDATOS_CSV, df_selection=current_selection)

            # Data for prediction
            st.session_state.patient_data = prepare_patient_data_for_prediction(
                extract_real_data(
                    ruta_archivo_csv=RUTA_FICHERODEDATOS_CSV, index=current_selection, return_type="tuple"
                )
            )

            # Build new DataFrame with Simulation - Prediction result
            # Assing to -session_state
            st.session_state.df_sim_datos_reales = build_df_for_stats(
                data,
                corridas_sim,
                patient_data=st.session_state.patient_data,
                include_mean=True,
                include_std=True,
                include_confint=True,
                include_metrics=True,
                include_prediction_mean=True,
                metrics_as_percentage=True,
                include_info_label=True,
            )

        st.session_state.prev_selection = current_selection

    # print(st.session_state.df_sim_datos_reales)

    # If Simulation & Prediction
    if not (st.session_state.df_sim_datos_reales.empty and st.session_state.patient_data is None):
        toggle_format = st.toggle(
            label=LABEL_TIME_FORMAT, value=False, help=HELP_MSG_TIME_FORMAT, key="formato-tiempo-datos-reales"
        )

        # Simulation render Dataframe
        if not toggle_format:
            st.dataframe(
                st.session_state.df_sim_datos_reales,
                hide_index=True,
                use_container_width=True,
            )
        else:
            st.dataframe(
                format_time_columns(df=st.session_state.df_sim_datos_reales, exclude_rows=["Métrica de Calibración"]),
                hide_index=True,
                use_container_width=True,
            )

        # Prediction metric
        try:
            prediction_df = get_data_for_prediction(data=st.session_state.patient_data)
            preds, preds_proba = predict(prediction_df)

            if "prediccion_datos_reales_clases" not in st.session_state:
                st.session_state.prediccion_datos_reales_clases = 0
            if "prediccion_datos_reales_porcentaje" not in st.session_state:
                st.session_state.prediccion_datos_reales_porcentaje = 0.0

            st.session_state.prediccion_datos_reales_clases = preds[0]
            st.session_state.prediccion_datos_reales_porcentaje = preds_proba[0]

            current_pred = float(st.session_state.prediccion_datos_reales_porcentaje)

            delta_label = ""
            delta_color = "normal"

            paciente_vive = st.session_state.prediccion_datos_reales_clases == 0
            metric_display_value = (
                f"{'Paciente no fallece' if paciente_vive else 'Paciente fallece'} ({(current_pred * 100):.0f}%)"
            )

            st.metric(
                label=LABEL_PREDICTION_METRIC,
                value=metric_display_value,
                delta=delta_label,
                delta_color=delta_color,
                border=True,
                width="stretch",
                help="Predicción basada en el modelo de machine learning entrenado con datos históricos",
            )

            st.session_state.prev_prediccion_datos_reales_porcentaje = current_pred

        except Exception as e:
            st.warning(f"No se pudo realizar la predicción: {e}")

    # SIMULAR TODOS LOS DATOS EN LA TABLA.


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
                file_upl1 = st.file_uploader(label="Experimento 1", type=[".csv"], accept_multiple_files=False)
            with col2:
                file_upl2 = st.file_uploader(label="Experimento 2", type=[".csv"], accept_multiple_files=False)

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
            EXP_VARIABLES,
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
                            st.dataframe(df_mostrar, hide_index=True, use_container_width=True)
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
            file_upl_experimentos = st.file_uploader(label="Experimentos", type=[".csv"], accept_multiple_files=True)
            dataframes_experimentos = bin_to_df(file_upl_experimentos)

        # Columna a comparar.
        opcion_col_comparacion = st.selectbox("Seleccione una columna para comparar", EXP_VARIABLES, key=3)
        boton_comparacion = st.button(
            "Realizar prueba de Friedman",
            type="primary",
            use_container_width=True,
            key=4,
        )

        with st.container():
            if boton_comparacion:
                if len(file_upl_experimentos) == 0:
                    st.warning("No se han cargado datos de resultados de experimentos para realizar esta prueba.")
                elif not len(file_upl_experimentos) >= 3:
                    st.warning("Debe cargar más de 3 muestras para realizar esta prueba.")
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
                        st.dataframe(df_mostrar, hide_index=True, use_container_width=True)
                        st.markdown(INFO_STATISTIC)
                        st.markdown(INFO_P_VALUE)
                    except Exception as data:
                        st.exception(data)
