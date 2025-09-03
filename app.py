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
    format_time_columns,
    generate_id,
    get_data_for_prediction,
    key_categ,
    predict,
    simulate_real_data,
    start_experiment,
    value_is_zero,
    prepare_patient_data_for_prediction,
    _extract_real_data,
    build_comprehensive_stats_table,
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
st.set_page_config(
    page_title="Simulación UCI - Análisis de Pacientes", page_icon="🏥", layout="wide", initial_sidebar_state="expanded"
)


# Theme configs
if "theme" not in st.session_state:
    st.session_state.theme = "light"

# Barra lateral con controles
with st.sidebar:
    st.title("Controles")

    # Toggle de modo oscuro con mejor diseño
    theme_toggle = st.toggle(
        "Modo Oscuro", value=st.session_state.theme == "dark", help="Cambiar entre modo claro y oscuro"
    )

    # Actualizar tema si cambió
    new_theme = "dark" if theme_toggle else "light"
    if new_theme != st.session_state.theme:
        st.session_state.theme = new_theme
        apply_theme(new_theme)
        st.rerun()

    st.divider()

    # Información adicional sobre la app
    st.markdown("### 📊 Sobre la App")
    st.markdown("""
    Esta aplicación permite:
    - Simular pacientes UCI
    - Analizar datos reales
    - Realizar predicciones
    - Comparar resultados estadísticos
    """)

    # Versión
    st.markdown("---")
    st.caption("Versión 2.0 - Septiembre 2025")

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
            opcion_edad: int = st.number_input(label="Edad", min_value=EDAD_MIN, max_value=EDAD_MAX, value=EDAD_DEFAULT)
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
        boton_comenzar_simulacion = st.button("Comenzar Simulación", type="primary", use_container_width=True)

    # Mostrar DataFrame con resultado de la simulación para ese paciente.
    if not st.session_state.df_resultado.empty:
        toggle_format = st.toggle(
            label=LABEL_TIME_FORMAT,
            value=True,
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
        corridas_sim = CORRIDAS_SIM_DEFAULT

        col1, col2 = st.columns(2)

        # with st.expander(expanded=True, width="stretch", label="Resultados de simulación con paciente seleccionado"):
        # with st.popover(label="Resultados de simulación con paciente seleccionado"):

        with col1:
            toggle_format = st.toggle(
                label=LABEL_TIME_FORMAT, value=False, help=HELP_MSG_TIME_FORMAT, key="formato-tiempo-datos-reales"
            )
        with col2:
            label: str = "Cantidad de Simulaciones por paciente"

            with st.popover(
                label=label,
                use_container_width=True,
                help=HELP_MSG_CORRIDA_SIM,
            ):
                corridas_sim = st.number_input(
                    label=label,
                    min_value=CORRIDAS_SIM_MIN,
                    max_value=CORRIDAS_SIM_MAX,
                    value=CORRIDAS_SIM_DEFAULT,
                    help=HELP_MSG_CORRIDA_SIM,
                )

        data: tuple[float] = simulate_real_data(ruta_fichero_csv=RUTA_FICHERODEDATOS_CSV, df_selection=selection)

        patient_tuple = _extract_real_data(RUTA_FICHERODEDATOS_CSV, selection, "tuple")
        patient_data = prepare_patient_data_for_prediction(patient_tuple)

        df_sim_datos_reales = build_df_for_stats(
            data,
            corridas_sim,
            include_mean=True,
            include_std=True,
            include_confint=True,
            include_metrics=True,
            include_prediction_mean=True,
            metrics_as_percentage=True,
            patient_data=patient_data,
            include_info_label=True,
        )

        if toggle_format:
            st.dataframe(
                format_time_columns(df_sim_datos_reales, exclude_rows=["Métrica de Calibración"]),
                hide_index=True,
                use_container_width=True,
            )
        else:
            st.dataframe(
                df_sim_datos_reales,
                hide_index=True,
                use_container_width=True,
            )

        # Mostrar predicción del paciente seleccionado
        try:
            prediction_df = get_data_for_prediction(patient_data)
            preds, preds_proba = predict(prediction_df)

            # Almacenar predicción en session_state
            if "prediccion_datos_reales_clases" not in st.session_state:
                st.session_state.prediccion_datos_reales_clases = 0
            if "prediccion_datos_reales_porcentaje" not in st.session_state:
                st.session_state.prediccion_datos_reales_porcentaje = 0.0

            st.session_state.prediccion_datos_reales_clases = preds[0]
            st.session_state.prediccion_datos_reales_porcentaje = preds_proba[0]

            # Mostrar métrica de predicción
            prev_pred = st.session_state.get("prev_prediccion_datos_reales_porcentaje", None)
            current_pred = float(st.session_state.prediccion_datos_reales_porcentaje)

            delta_label = ""
            delta_color = "normal"

            if prev_pred is not None:
                try:
                    prev_val = float(prev_pred)
                    change = current_pred - prev_val
                    percent_change = change * 100

                    if change > 0:
                        how_changed = "Incremento"
                        delta_color = "inverse"
                    elif change < 0:
                        how_changed = "Disminución"
                        delta_color = "normal"
                    else:
                        delta_label = "Sin cambio"
                        delta_color = "off"

                    if change != 0:
                        delta_label = f"{how_changed} de {abs(percent_change):.0f}% en la probabilidad de fallecimiento"

                except Exception:
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

    st.divider()

    # Simular todos los datos en la tabla.
    if st.button(
        "Simular y evaluar todos los datos en la tabla",
        type="primary",
        use_container_width=True,
        key="simular_tabla",
    ):
        with st.spinner(
            text="Simulando todos los datos en la tabla y calculando predicciones. Esto puede tardar varios minutos...",
            show_time=True,
        ):
            try:
                # Crear barra de progreso
                progress_bar = st.progress(0)

                # Función callback para actualizar progreso
                def update_progress(progress):
                    progress_bar.progress(progress)

                # Usar la nueva función para tabla comprehensiva
                df_pacientes, df_general, df_calibracion, friedman_results, messages = build_comprehensive_stats_table(
                    RUTA_FICHERODEDATOS_CSV, corridas_sim, progress_callback=update_progress
                )

                # Limpiar barra de progreso
                progress_bar.empty()

                if not df_general.empty:
                    # Almacenar en session_state
                    st.session_state.df_pacientes_individuales = df_pacientes
                    st.session_state.df_sim_datos_reales = df_general
                    st.session_state.df_calibracion = df_calibracion
                    st.session_state.friedman_results = friedman_results

                    # Mostrar mensajes informativos
                    for msg in messages["info"]:
                        st.info(msg)

                    # Mostrar warnings si los hay
                    for warning in messages["warnings"]:
                        st.warning(warning)

                    # Mostrar errores si los hay
                    for error in messages["errors"]:
                        st.error(error)

                    st.toast(
                        f"Se ha completado la simulación y predicción para {len(pd.read_csv(RUTA_FICHERODEDATOS_CSV))} pacientes. Se muestran resultados individuales y promedios generales.",
                        icon="✅",
                    )
                else:
                    st.error("No se pudo generar la tabla de resultados.")

            except Exception as e:
                st.toast(
                    f"Se ha producido un error al realizar la simulación: {e}",
                    icon="⚠️",
                )

    # Mostrar simulación con datos reales.
    if (
        "df_sim_datos_reales" in st.session_state
        and hasattr(st.session_state.df_sim_datos_reales, "empty")
        and not st.session_state.df_sim_datos_reales.empty
        and "df_pacientes_individuales" in st.session_state
        and hasattr(st.session_state.df_pacientes_individuales, "empty")
        and not st.session_state.df_pacientes_individuales.empty
        and "df_calibracion" in st.session_state
        and hasattr(st.session_state.df_calibracion, "empty")
        and not st.session_state.df_calibracion.empty
    ):
        st.subheader("📊 Estadísticas Comprehensivas de Simulación y Predicción")

        # Mostrar tabla de pacientes individuales
        if "df_pacientes_individuales" in st.session_state:
            st.subheader("📋 Resultados Individuales por Paciente")
            st.dataframe(
                st.session_state.df_pacientes_individuales,
                hide_index=True,
                use_container_width=True,
            )
            st.info("Esta tabla muestra el promedio de simulación para cada paciente individual.")

        # Mostrar tabla de calibración
        if "df_calibracion" in st.session_state:
            st.subheader("🎯 Métricas de Calibración por Paciente")
            st.dataframe(
                st.session_state.df_calibracion,
                hide_index=True,
                use_container_width=True,
            )
            st.info(
                "Esta tabla muestra qué porcentaje de las simulaciones de cada paciente están dentro del intervalo de confianza basado en sus valores reales."
            )

        # Mostrar tabla de promedio general
        st.subheader("📈 Promedio General de Todos los Pacientes")
        display_df_reales = st.session_state.df_sim_datos_reales.copy()

        st.dataframe(
            display_df_reales,
            hide_index=False,
            use_container_width=True,
        )

        st.info(
            "📈 **Interpretación:** La primera tabla muestra resultados individuales por paciente. La segunda tabla muestra métricas de calibración. La tercera tabla muestra el promedio general de todos los pacientes."
        )

        csv_sim_datos_reales = st.session_state.df_sim_datos_reales.to_csv(index=True).encode("UTF-8")

        # st.download_button(
        #     label="💾 Guardar resultados comprehensivos",
        #     data=csv_sim_datos_reales,
        #     file_name="Estadisticas_Comprehensivas_Simulacion_Prediccion.csv",
        #     mime="text/csv",
        #     use_container_width=True,
        #     key="guardar_sim_datos_reales",
        # )

        # csv_pacientes_individuales = st.session_state.df_pacientes_individuales.to_csv(index=False).encode("UTF-8")

        # st.download_button(
        #     label="💾 Guardar resultados individuales por paciente",
        #     data=csv_pacientes_individuales,
        #     file_name="Resultados_Individuales_Pacientes.csv",
        #     mime="text/csv",
        #     use_container_width=True,
        #     key="guardar_pacientes_individuales",
        # )

        # csv_calibracion = st.session_state.df_calibracion.to_csv(index=False).encode("UTF-8")

        # st.download_button(
        #     label="💾 Guardar métricas de calibración",
        #     data=csv_calibracion,
        #     file_name="Metricas_Calibracion_Pacientes.csv",
        #     mime="text/csv",
        #     use_container_width=True,
        #     key="guardar_calibracion",
        # )

        # Mostrar resultados del test de Friedman
        if "friedman_results" in st.session_state and st.session_state.friedman_results:
            st.subheader("🧪 Test de Friedman - Comparación entre Pacientes")

            st.markdown("""
            **Interpretación del Test de Friedman:**
            - **Estadístico F:** Valor del test estadístico
            - **Valor p:** Probabilidad de que las diferencias sean aleatorias
            - **Significativo:** Si p < 0.05, hay diferencias significativas entre los pacientes
            """)

            # Crear tabla con resultados de Friedman
            friedman_data = []
            for var, results in st.session_state.friedman_results.items():
                if results.get("error"):
                    friedman_data.append(
                        {
                            "Variable": var,
                            "Estadístico F": "N/A",
                            "Valor p": "N/A",
                            "Significativo": "Error",
                        }
                    )
                else:
                    friedman_data.append(
                        {
                            "Variable": var,
                            "Estadístico F": f"{results.get('statistic', 'N/A'):.4f}"
                            if results.get("statistic") is not None
                            else "N/A",
                            "Valor p": f"{results.get('p_value', 'N/A'):.4f}"
                            if results.get("p_value") is not None
                            else "N/A",
                            "Significativo": "Sí" if results.get("significant") else "No",
                        }
                    )

            friedman_df = pd.DataFrame(friedman_data)
            st.dataframe(
                friedman_df,
                hide_index=True,
                use_container_width=True,
                column_config={
                    "Variable": st.column_config.TextColumn("Variable", width="medium"),
                    "Estadístico F": st.column_config.TextColumn("Estadístico F", width="small"),
                    "Valor p": st.column_config.TextColumn("Valor p", width="small"),
                    "Significativo": st.column_config.TextColumn("Significativo", width="small"),
                },
            )

            # Resumen de resultados significativos
            significant_vars = [
                var
                for var, results in st.session_state.friedman_results.items()
                if results.get("significant") and not results.get("error")
            ]

            if significant_vars:
                st.success(f"📈 **Variables con diferencias significativas:** {', '.join(significant_vars)}")
            else:
                st.info(
                    "📊 **No se encontraron diferencias significativas entre los pacientes para ninguna variable.**"
                )

            # Información adicional
            st.info("""
            **¿Qué significa esto?**
            - Si una variable tiene p < 0.05, significa que hay diferencias estadísticamente significativas entre los pacientes
            - Esto indica que los pacientes responden de manera diferente a las simulaciones
            - Las predicciones también pueden variar significativamente entre pacientes
            """)

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
