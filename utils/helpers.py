import secrets
import random
from typing import List, Tuple

import numpy as np
import pandas as pd
from pandas import DataFrame
from streamlit.runtime.uploaded_file_manager import UploadedFile
import streamlit as st

from uci.stats import StatsUtils
from utils.constants import (
    AGE_MIN,
    FICHERODEDATOS_CSV_PATH,
    SIM_RUNS_DEFAULT,
    EXPERIMENT_VARIABLES_LABELS as EXP_VARS,
    PREDICTION_MODEL_PATH,
    VENTILATION_TYPE,
    PREUCI_DIAG,
    RESP_INSUF,
)

import joblib

import sys
import traceback


def key_categ(category: str, value: str | int, viceversa: bool = False) -> int | str:
    """
    Return the key corresponding to a given value in the category mappings defined in `utils.constants`.

    Args:
        category: One of the category identifiers: "va", "diag" or "insuf".
        value: The value to look up (or the key to look up if `viceversa` is True).
        viceversa: If False (default) the function searches for the key that maps to `value`.
                   If True the function treats `value` as a key and returns its mapped value.

    Returns:
        The matching key (int) when searching by value, or the matching value (str) when
        searching by key (viceversa=True).

    Raises:
        Exception: If the category is unknown or no match is found.
    """

    categorias: dict[int, str]

    if category == "vt":
        categorias = VENTILATION_TYPE
    elif category == "diag":
        categorias = PREUCI_DIAG
    elif category == "insuf":
        categorias = RESP_INSUF
    else:
        raise Exception(f"The selected category does not exist: {category}.")

    for k, v in categorias.items():
        if not viceversa:
            if v == value:
                return k
        else:
            if k == value:
                return v

    if not viceversa:
        raise Exception(f"The provided value was not found in the category set: {category}.")
    else:
        raise Exception(f"The provided key was not found in the category set: {category}.")


def value_is_zero(values: list[int | str] | int | str) -> bool:
    """
    Check whether a value or all values in a list are zero (or the Spanish string "Vacío").

    Args:
        values: A single value or a list of values to check.

    Returns:
        True if the given value(s) are 0 or the string "Vacío" (case-insensitive), False otherwise.
    """

    def __iszero(v: int | str) -> bool:
        if isinstance(v, int):
            return v == 0
        elif isinstance(v, str):
            return v.lower() == "vacío"
        return False

    if isinstance(values, (int, str)):
        return __iszero(values)
    elif isinstance(values, list):
        return all(__iszero(v) for v in values)
    return False


def generate_id(digits: int = 10) -> str:
    """
    Generate a pseudo-random numeric identifier with the specified number of digits.

    Args:
        digits: Number of digits in the generated ID (1..10). Default is 10.

    Returns:
        A string containing the generated numeric ID.
    """

    if not 0 < digits <= 10:
        raise Exception(f"The number of digits n={digits} must be in the range 0 < n <= 10.")
    # Use randbelow(10) to generate each digit (0-9)
    return "".join([str(secrets.randbelow(10)) for _ in range(digits)])


def format_df_time(df: DataFrame, rows_to_format: list[int] = None) -> DataFrame:
    """
    Return a copy of the DataFrame with numeric time values prepared for display.

    The DataFrame values remain numeric to preserve serializability; formatting is applied
    to a copy for presentation purposes.

    Args:
        df: DataFrame to transform.
        rows_to_format: Optional list of row indices to format. If None, all rows are formatted.

    Returns:
        A DataFrame copy with formatted values for display.
    """

    # Make a copy of the DataFrame to avoid modifying the original
    result_df = df.copy()

    # If rows_to_format is not specified, apply to all rows
    if rows_to_format is None:
        rows_to_format = list(range(len(df)))

    # Apply formatting only to the specified rows
    for idx, row in df.iterrows():
        if idx in rows_to_format:
            for col in df.columns:
                try:
                    # Try to convert to float
                    value = float(row[col])
                    if not pd.isna(value):
                        # Keep the original numeric value
                        # Formatting will be applied only for display
                        result_df.at[idx, col] = value
                except (ValueError, TypeError):
                    # If it cannot be converted to float, keep the original value
                    pass

    return result_df


def format_time_columns(
    df: DataFrame,
    exclude_rows: list[str] = None,
) -> DataFrame:
    """
    Format columns that represent time (hours) for display.

    Rows whose label (in the 'Información' column or index name) is present in
    `exclude_rows` will not be reformatted.

    Args:
        df: DataFrame containing the data to format.
        exclude_rows: Optional list of row labels to exclude from formatting.

    Returns:
        A DataFrame with the time columns formatted as human-readable strings.
    """
    if exclude_rows is None:
        exclude_rows = []

    # Make a copy to avoid mutating the original
    result_df = df.copy()

    # Columns that typically contain time values
    time_columns = [
        "Tiempo Pre VAM",
        "Tiempo VAM",
        "Tiempo Post VAM",
        "Estadia UCI",
        "Estadia Post UCI",
    ]

    # Filter only the columns that exist in the DataFrame
    time_columns = [col for col in time_columns if col in result_df.columns]

    # If there are no time columns, return the DataFrame unchanged
    if not time_columns:
        return result_df

    # Check whether the DataFrame contains an 'Información' column
    has_info_column = "Información" in result_df.columns

    # Apply formatting to time columns
    for col in time_columns:
        # Create a copy of the column and change its type to 'object' to avoid FutureWarning
        formatted_col = result_df[col].astype(object)

        # If there is an 'Información' column, find rows to exclude
        if has_info_column:
            for idx in result_df.index:
                # Check whether the current row is in the exclusion list
                if result_df.at[idx, "Información"] not in exclude_rows:
                    try:
                        value = result_df.at[idx, col]
                        if pd.notna(value):
                            try:
                                # Convert to string to avoid type issues
                                formatted_col.at[idx] = format_value_for_display(value)
                            except (ValueError, TypeError) as e:
                                print(
                                    f"Error formatting value {value} in column '{col}', row {idx} (Información: {result_df.at[idx, 'Información']}): {str(e)}"
                                )
                    except Exception as e:
                        print(f"Unexpected error accessing data in column '{col}', row {idx}: {str(e)}")
        # If there is no 'Información' column but the index name is 'Información'
        elif result_df.index.name == "Información":
            for idx in result_df.index:
                if idx not in exclude_rows:
                    try:
                        value = result_df.at[idx, col]
                        if pd.notna(value):
                            try:
                                # Convert to string to avoid type issues
                                formatted_col.at[idx] = format_value_for_display(value)
                            except (ValueError, TypeError) as e:
                                print(f"Error formatting value {value} in column '{col}', index '{idx}': {str(e)}")
                    except Exception as e:
                        print(f"Unexpected error accessing data in column '{col}', index '{idx}': {str(e)}")
        else:
            # Para DataFrames normales, formatear todas las filas
            formatted_col = result_df[col].apply(lambda x: format_value_for_display(x) if pd.notna(x) else x)

        # Assign the formatted column back, converting everything to string to avoid ArrowTypeError
        result_df[col] = formatted_col.astype(str)

    return result_df


def format_value_for_display(value: float | int) -> str:
    """
    Format a numeric value (in hours) into a human-readable string showing days and hours/minutes.

    Examples: "1.0 d (24.0 h)" or "0.5 d (30 min)".

    Args:
        value: Numeric value in hours.

    Returns:
        A formatted string representation of the value.
    """

    try:
        value = float(value)
        if pd.isna(value):
            return str(value)

        if value >= 1.0:
            return f"{value / 24:.1f} d ({value:.1f} h)"
        else:
            return f"{value / 24:.1f} d ({value * 60:.0f} min)"
    except (ValueError, TypeError):
        return str(value)


def format_df_stats(
    df: DataFrame,
    column_label: str = "Información",
    labels_structure: dict[int | str] | list[str] | None = None,
) -> DataFrame:
    """
    Insert a left-most column with labels for the provided DataFrame.

    If `labels_structure` is provided it may be a dict (index->label) or a list of labels.
    When `labels_structure` is None, sensible defaults are applied:
      - If the DataFrame has a RangeIndex and multiple rows, unlabeled rows are named "Paciente {i}".
      - Otherwise unlabeled rows receive the string "---".

    Args:
        df: DataFrame to modify.
        column_label: Name of the inserted label column (default: "Información").
        labels_structure: Optional dict or list to drive row labels.

    Returns:
        The DataFrame with the new left-most label column.
    """

    # Insert the column at the beginning
    df.insert(0, column_label, "")

    unassigned_label = "---"

    n_rows = df.shape[0]

    # Normalize labels_structure to a dict
    labels_map: dict[int, str] = {}
    if labels_structure is None:
        labels_map = {}
    elif isinstance(labels_structure, dict):
        # convert keys to int if possible
        for k, v in labels_structure.items():
            try:
                idx = int(k)
            except Exception:
                continue
            labels_map[idx] = str(v)
    elif isinstance(labels_structure, list):
        for i, v in enumerate(labels_structure):
            labels_map[i] = str(v)

    # Assign provided labels
    for idx, label in labels_map.items():
        if 0 <= idx < n_rows:
            df.at[idx, column_label] = label

    # Fill rows without a label
    for i in range(n_rows):
        if df.at[i, column_label] == "":
            # If it looks like a multi-patient DataFrame (rows > 1 and RangeIndex), use 'Paciente i'
            if n_rows > 1 and isinstance(df.index, pd.RangeIndex):
                df.at[i, column_label] = f"Paciente {i}"
            else:
                df.at[i, column_label] = unassigned_label

    return df


def build_df_for_stats(
    data: DataFrame | list[DataFrame],
    sample_size: int | None = None,
    include_mean=True,
    include_std=False,
    include_confint=False,
    include_metrics=False,
    include_prediction_mean=False,
    include_info_label=True,
    labels_structure: dict[int | str] | list[str] | None = None,
    metrics_as_percentage: bool = False,
    metrics_reference: pd.Series | dict | None = None,
    patient_data: dict | None = None,
    precise_values=True,
) -> DataFrame:
    """
    Build a summary DataFrame with statistics (mean, std, confidence intervals, calibration metrics)
    from either a vertical DataFrame or a list of DataFrames (one DataFrame per patient).

    Args:
        - data: DataFrame or list[DataFrame].
        - sample_size: Required when include_confint=True.
        - include_*: Flags to include mean, std, confint, calibration metrics, prediction mean.
        - include_info_label: Add the 'Información' label column when True.
        - labels_structure: Optional dict {index: label} or list of labels to apply.
        - patient_data: Optional dict with patient features required for prediction (when include_prediction_mean=True).

    Notes:
        - Confidence intervals require mean, std and sample_size>0.
        - If labels_structure is not provided labels are inferred based on included statistics.
    """

    # Minimal robust implementation.
    column_label = "Información"

    # Minimal validations.
    if include_confint and (not include_mean or not include_std):
        raise ValueError("Calculating confidence intervals requires include_mean=True and include_std=True.")
    if include_confint and (sample_size is None or sample_size <= 0):
        raise ValueError("`sample_size` must be > 0 when include_confint=True.")

    # If a list of DataFrames is provided: produce one row per patient with means
    if isinstance(data, list):
        if not data:
            return pd.DataFrame()

        rows = []
        for df_i in data:
            # Take means of the expected columns; if columns are missing pandas will fill with NaN
            rows.append(df_i[EXP_VARS].mean())

        df_output = pd.DataFrame(rows).reset_index(drop=True)

        if include_info_label:
            # If explicit labels are provided, they are applied; otherwise format_df_stats will fill 'Patient i'
            df_output = format_df_stats(df_output, column_label=column_label, labels_structure=labels_structure)

        return df_output

    # If it's a single DataFrame: build vertical rows according to flags
    if isinstance(data, DataFrame):
        df = data
        rows = []
        auto_labels: list[str] = []

        #########
        # MEAN #
        #########
        if include_mean:
            rows.append(df[EXP_VARS].mean())
            auto_labels.append("Promedio")

        ##################
        # STANDARD DEV #
        ##################
        if include_std:
            rows.append(df[EXP_VARS].std())
            auto_labels.append("Desviación Estándar")

        ##########################
        # CONFIDENCE INTERVALS #
        ##########################
        if include_confint:
            mean = df[EXP_VARS].mean()
            std = df[EXP_VARS].std()

            li, ls = StatsUtils.confidenceinterval(mean.values, std.values, sample_size)
            li_s = pd.Series(li, index=EXP_VARS)
            ls_s = pd.Series(ls, index=EXP_VARS)
            rows.append(li_s)
            auto_labels.append("Límite Inf")
            rows.append(ls_s)
            auto_labels.append("Límite Sup")

        ###########################
        # CALIBRATION METRICS #
        ###########################
        if include_metrics:
            # Calibration metric: count how many iterations fell inside the [LI, LS] interval.

            # If `metrics_reference` is provided as Series/dict with real values, real coverage is evaluated -> CI.

            # If `metrics_as_percentage=True`, return percentage instead of counts.

            if include_confint:
                try:
                    counts = {}
                    for col in EXP_VARS:
                        lower = li_s[col]
                        upper = ls_s[col]

                        if metrics_reference is not None:
                            # If metrics_reference has a scalar for the column -> 0/1 if it lies within the interval
                            if isinstance(metrics_reference, pd.Series) or isinstance(metrics_reference, dict):
                                ref_val = (
                                    metrics_reference.get(col)
                                    if isinstance(metrics_reference, dict)
                                    else metrics_reference.get(col, None)
                                )
                                if ref_val is None:
                                    counts[col] = 0
                                else:
                                    counts[col] = int(lower <= ref_val <= upper)
                            else:
                                # Not supported, use placeholder
                                counts[col] = 0
                        else:
                            # Count rows in the original DataFrame that are inside the interval
                            counts[col] = int(((df[col] >= lower) & (df[col] <= upper)).sum())

                    metrics_row = pd.Series([counts.get(col, 0) for col in EXP_VARS], index=EXP_VARS)

                    # If percentage is requested and we work with iteration counts
                    if metrics_as_percentage and metrics_reference is None:
                        denom = df.shape[0] if df.shape[0] > 0 else 1
                        metrics_row = metrics_row.astype(float) / denom * 100.0
                    elif metrics_as_percentage and metrics_reference is not None:
                        # Cuando comparamos con una referencia escalar, porcentaje es 0 o 100
                        metrics_row = metrics_row.astype(float) * 100.0

                except Exception:
                    # In case of any failure, use placeholder.
                    metrics_row = pd.Series([0] * len(EXP_VARS), index=EXP_VARS)
            else:
                # If no interval was calculated, counting coverage is meaningless — use placeholder.
                metrics_row = pd.Series([0] * len(EXP_VARS), index=EXP_VARS)

            rows.append(metrics_row)
            auto_labels.append("Calibration Metric")

        if not rows:
            raise ValueError("At least one statistic must be included (mean/std/confint/metrics).")

        df_output = pd.DataFrame(rows).reset_index(drop=True)

        if include_info_label:
            # If labels_structure is provided apply it; otherwise use auto_labels for a single patient
            labels_to_use = labels_structure if labels_structure is not None else auto_labels
            df_output = format_df_stats(df_output, column_label=column_label, labels_structure=labels_to_use)

        return df_output

    raise TypeError("`data` must be a DataFrame or a list of DataFrames.")


def bin_to_df(files: UploadedFile | list[UploadedFile]) -> DataFrame | list[DataFrame]:
    """
    Convert one or more uploaded files (Streamlit UploadedFile) into DataFrame(s).

    Args:
        files: A single UploadedFile or a list of UploadedFile objects.

    Returns:
        A DataFrame (for single file) or a list of DataFrames (for multiple files).
    """

    if isinstance(files, UploadedFile):
        return pd.read_csv(files)
    elif isinstance(files, list):
        return [pd.read_csv(f) for f in files]


def extract_true_data_from_csv(
    csv_path: str,
    index: int | None = None,
    as_dataframe=True,
    **kwargs,
) -> DataFrame | tuple[float]:
    """
    Extract the clinical data for a single patient or all patients from a CSV by row index.

    The CSV should contain columns: edad, apache, tiempo_vam, estadia_uci, diagnostico_preuci, tipo_causavam, etc.

    Args:
        csv_path: Path to the CSV file with the real data.
        index: Integer row index of the patient to extract. If None, extract all patients.
        as_dataframe: If True, return DataFrame(s). If False, return dict(s).

    Returns:
        For single patient (index provided):
            - DataFrame with one row (when as_dataframe=True)
            - Dict with patient values (when as_dataframe=False)
        For all patients (index=None):
            - DataFrame with all patients (when as_dataframe=True)
            - List of dicts, one per patient (when as_dataframe=False)
    """

    # Backwards compatibility: accept Spanish kwarg 'ruta_archivo_csv'
    if "ruta_archivo_csv" in kwargs and (csv_path is None or csv_path == ""):
        csv_path = kwargs.get("ruta_archivo_csv")

    data: pd.DataFrame = pd.read_csv(csv_path)

    # Reuse top-level builder if available
    def build_row_local(data_index: int):
        return build_row_from_dataframe(data, data_index)

    if index is not None:
        extracted_data = build_row_local(index)
    else:
        # Extract data for all patients
        extracted_data = []
        for i in range(len(data)):
            try:
                row_data = build_row_local(i)
                extracted_data.append(row_data)
            except (IndexError, KeyError, ValueError) as e:
                print(f"Warning: Could not extract data for patient at index {i}: {e}")
                continue

    # Backwards-compatible support for callers that request a tuple
    # via return_type='tuple' (legacy behavior expected by some helpers).
    return_type = kwargs.get("return_type", None)

    if return_type == "tuple":
        # Map the dict produced by build_row_from_dataframe into the expected
        # tuple order used across the codebase:
        # (edad, d1, d2, d3, d4, apache, insuf, va, est_uci, tiempo_vam, est_preuti, diag_egr2)
        def dict_to_tuple(d: dict) -> tuple:
            return (
                int(d.get("edad", AGE_MIN)),
                int(d.get("d1", 0)),
                int(d.get("d2", 0)),
                int(d.get("d3", 0)),
                int(d.get("d4", 0)),
                int(d.get("apache", 0)),
                int(d.get("insuf", 0)),
                int(d.get("va", 0)),
                int(d.get("estuci", 0)),
                int(d.get("tiempo_vam", 0)),
                int(d.get("estpreuci", 0)),
                int(d.get("diag_egr2", 0)),
            )

        if index is not None:
            return dict_to_tuple(extracted_data)
        else:
            return [dict_to_tuple(d) for d in extracted_data]

    if as_dataframe:
        if index is not None:
            # Single patient - return DataFrame with one row
            return pd.DataFrame([extracted_data])
        else:
            # All patients - return DataFrame with multiple rows
            return pd.DataFrame(extracted_data)
    else:
        if index is not None:
            # Single patient - return dict of values
            return extracted_data
        else:
            # All patients - return list of dicts
            return extracted_data


def build_row_from_dataframe(data: pd.DataFrame, data_index: int) -> dict:
    """
    Build the standardized patient dict from a DataFrame row. This is the extracted
    logic previously embedded inside `extract_true_data_from_csv`.
    """
    # estuci: days -> hours
    # tiempo_vam: horas
    # estpreuci: days -> hours

    # Safely fetch values and provide defaults when missing
    def _safe(col, default=0):
        try:
            v = data[col].iloc[data_index]
            if pd.isna(v):
                return default
            return v
        except Exception:
            return default

    diag_egr2_val = _safe("Diag.Egr2", 0)

    return {
        "edad": int(_safe("Edad", AGE_MIN)),
        "d1": int(_safe("Diag.Ing1", 0)),
        "d2": int(_safe("Diag.Ing2", 0)),
        "d3": int(_safe("Diag.Ing3", 0)),
        "d4": int(_safe("Diag.Ing4", 0)),
        "apache": int(_safe("APACHE", 0)),
        "insuf": int(_safe("InsufResp", 0)),
        "va": int(_safe("VA", 0)),
        "estuci": int(_safe("Est. UCI", 0) * 24),
        "tiempo_vam": int(_safe("TiempoVAM", 0)),
        "estpreuci": int(_safe("Est. PreUCI", 0) * 24),
        "diag_egr2": int(diag_egr2_val),
    }


def run_experiment(
    n_runs: int,
    age: int,
    d1: int,
    d2: int,
    d3: int,
    d4: int,
    apache: int,
    resp_insuf: int,
    artif_vent: int,
    vam_time: int,
    uti_stay: int,
    preuti_stay: int,
    percent: int = 10,
) -> pd.DataFrame:
    """
    Run the simulation for a single patient using the provided clinical inputs.

    Args:
        n_runs: Number of simulation replications to execute.
        age: Patient age.
        d1-d4: Admission diagnoses (diagnostic codes).
        apache: APACHE score.
        resp_insuf: Respiratory insufficiency code.
        artif_vent: Ventilation/artificial ventilation indicator.
        vam_time: Ventilation type/time code.
        uti_stay: Expected ICU stay.
        preuti_stay: Pre-ICU stay.
        percent: Percentage parameter used by the experiment (default=10).

    Returns:
        DataFrame containing the simulation results with columns:
        ["Tiempo Pre VAM", "Tiempo VAM", "Tiempo Post VAM", "Estadia UCI", "Estadia Post UCI"].
    """

    # Local import to avoid circular import between uci.experiment and uci.simulacion
    from uci.experiment import Experiment, multiple_replication

    e = Experiment(
        age=age,
        diagnosis_admission1=d1,
        diagnosis_admission2=d2,
        diagnosis_admission3=d3,
        diagnosis_admission4=d4,
        apache=apache,
        respiratory_insufficiency=resp_insuf,
        artificial_ventilation=artif_vent,
        uti_stay=uti_stay,
        vam_time=vam_time,
        preuti_stay_time=preuti_stay,
        percent=percent,
    )

    # Run the simulation
    res = multiple_replication(e, n_runs)

    # Ensure all columns are numeric
    for col in res.columns:
        # Convert to numeric, forcing non-numeric values to NaN
        res[col] = pd.to_numeric(res[col], errors="coerce")

        # Fill NaN with 0 and convert to integer
        res[col] = res[col].fillna(0).astype("int64")

    # Verify there are no NaN or None values
    if res.isnull().values.any():
        # Instead of st.warning, simply fill with 0
        res = res.fillna(0)

    # Ensure the index is sequential
    res = res.reset_index(drop=True)

    return res


def adjust_df_sizes(dataframes: List[DataFrame]) -> Tuple[List[DataFrame], int]:
    """
    Return a tuple containing a list of DataFrames trimmed to the length of the shortest DataFrame
    in the input list, and the integer length of that shortest DataFrame.

    Args:
        dataframes: List of DataFrames.

    Returns:
        Tuple (list_of_dataframes, min_len). The integer is -1 if trimming was not necessary.

    """

    def all_equal(lst) -> bool | None:
        return None if not lst else len(set(lst)) == 1

    df_sizes: list[int] = [df.shape[0] for df in dataframes]

    try:
        if all_equal(df_sizes):
            return dataframes, -1
        else:
            min_len = min(df_sizes)
            return [df.head(min_len) for df in dataframes], min_len
    except Exception as e:
        print(f"Error adjusting DF sizes: {e}")


def build_df_test_result(statistic: float, p_value: float) -> DataFrame:
    """
    Build a simple DataFrame with statistic and p-value for displaying test results.

    Args:
        statistic: Test statistic value.
        p_value: P-value.

    Returns:
        DataFrame with the provided statistic and p-value.
    """

    S = "Estadístico"
    P = "Valor de P"

    data = {S: [statistic], P: [p_value]}
    df = pd.DataFrame(data)

    return df


def simulate_true_data(csv_path: str, selection: int, **kwargs) -> DataFrame | list[DataFrame]:
    """
    Run simulations using real patient data from a CSV file.

    Args:
        csv_path: Path to the CSV file with the real patient data.
        selection: Row index of the selected patient; use -1 to process all patients.

    Returns:
        A DataFrame with simulation results for a single patient or a list of DataFrames
        (one per patient) when selection == -1. Each DataFrame contains columns:
        ["Tiempo Pre VAM", "Tiempo VAM", "Tiempo Post VAM", "Estadia UCI", "Estadia Post UCI"].
    """

    # Backwards compatibility: accept 'ruta_fichero_csv' and 'df_selection' kwarg names
    if "ruta_fichero_csv" in kwargs and (csv_path is None or csv_path == ""):
        csv_path = kwargs.get("ruta_fichero_csv")
    if "df_selection" in kwargs and (selection is None or selection == 0):
        selection = kwargs.get("df_selection")

    def experiment_helper(t: tuple[float]) -> DataFrame:
        """Build a DataFrame with simulation results for a single patient's tuple `t`.

        Expected tuple order produced by `extract_real_data(..., return_type='tuple')`:
        (edad, d1, d2, d3, d4, apache, insuf, va, est_uci, tiempo_vam, est_preuti, diag_egr2)
        """

        # Allow callers to pass the legacy 'corridas_simulacion' kwarg to control runs
        n_runs = kwargs.get("corridas_simulacion", SIM_RUNS_DEFAULT)

        e = run_experiment(
            n_runs,
            age=int(t[0]),
            d1=int(t[1]),
            d2=int(t[2]),
            d3=int(t[3]),
            d4=int(t[4]),
            apache=int(t[5]),
            resp_insuf=int(t[6]),
            artif_vent=int(t[7]),
            vam_time=int(t[9]),
            uti_stay=int(t[8]),
            preuti_stay=int(t[10]),
            percent=random.randint(0, 10),
        )

        return e

    if selection != -1:
        t: tuple[float] = extract_true_data_from_csv(csv_path, index=selection, return_type="tuple")

        # Return a DataFrame with the simulation results
        return experiment_helper(t)
    elif selection == -1:
        datalen = pd.read_csv(csv_path).shape[0]

        # Return a list of DataFrames (one per patient)
        return [
            experiment_helper(extract_true_data_from_csv(csv_path, index=i, return_type="tuple"))
            for i in range(datalen)
        ]
    else:
        raise ValueError("The parameter df_selection must be -1 or a non-negative integer.")


def fix_seed(seed: int | None = None):
    """Set the seed for numpy and Python's random module. This is useful for simulations that use system-based random seeds. By setting the seed, you can obtain specific reproducible results. If the seed is None, it restores the random seed to its default state. It also sets Python's default random seed.

    Args:
        seed (int): Maximum value 2^32 - 1 (numpy)
        seed (None): If None, the seed returns to being random

    Raises:
        ValueError: If the maximum seed size is exceeded (uint32: 2^32 - 1)
        ValueError: If the seed is negative
    """

    try:
        import random

        if seed is not None:
            if seed > np.iinfo(np.int32).max:
                raise ValueError("Maximum permissible seed size exceeded (2^32 - 1)")
            if seed < 0:
                raise ValueError("Seed must be a non-negative integer (uint32)")
            random.seed(seed)
            np.random.seed(seed)
        else:
            random.seed(None)
            np.random.seed(None)
    except Exception as e:
        print(f"Error while setting the seed: {e}")


def predict(df: DataFrame) -> tuple[np.ndarray, np.ndarray]:
    """
    Perform prediction using a previously trained model saved in 'prediction_model.joblib'.

    Args:
        df (DataFrame): Input DataFrame for prediction. Expected columns:
            {'Diag.Ing1', 'Diag.Ing2', 'Diag.Egr2', 'TiempoVAM', 'APACHE', 'Edad'}

    Returns:
        tuple[np.ndarray, np.ndarray]: (predictions, positive_class_probabilities_rounded)

    Example:
    >>> preds, preds_proba = predict(df)
    >>> print(preds)
    [1, 0, 1]
    >>> print(preds_proba)
    [0.85, 0.12, 0.97]

    Raises:
        FileNotFoundError: If the model file cannot be found.
        Exception: If an error occurs during prediction.
    """

    try:
        # 8/26/2025 - model trained with sklearn 1.6.1
        model = joblib.load(PREDICTION_MODEL_PATH)

        preds = model.predict(df)
        preds_proba = model.predict_proba(df)
        res = (preds, np.round(preds_proba[:, 1], 2))

        return res
    except Exception as e:
        # Raise an exception instead of using Streamlit UI calls
        raise Exception(f"Error during prediction: {e}")


def get_data_for_prediction(data: dict[str, int] | pd.DataFrame) -> pd.DataFrame:
    """
    Build a DataFrame with the fields required by the prediction model.

    Args:
        data (dict or DataFrame): If dict, expected keys are 'Edad', 'Diag.Ing1', 'Diag.Ing2',
            'Diag.Egr2', 'TiempoVAM', 'APACHE'. If a DataFrame is provided, it must contain
            the required columns.

    Returns:
        pd.DataFrame with a single row ready for prediction.

    Raises:
        Exception: If an error occurs while constructing the DataFrame; traceback is printed
    """

    # NOTE Required: {'Diag.Ing2', 'Diag.Egr2', 'Diag.Ing1', 'TiempoVAM', 'APACHE', 'Edad'}

    try:
        if isinstance(data, pd.DataFrame):
            required_cols = [
                "Edad",
                "Diag.Ing1",
                "Diag.Ing2",
                "Diag.Egr2",
                "TiempoVAM",
                "APACHE",
            ]
            if not all(col in data.columns for col in required_cols):
                raise ValueError(f"DataFrame must contain the following columns: {', '.join(f'{required_cols}')}")
            return data[required_cols]

        elif isinstance(data, dict):
            diag_ing2 = data.get("Diag.Ing2", 0)
            diag_egr2 = data.get("Diag.Egr2", 0)
            diag_ing1 = data.get("Diag.Ing1", 0)
            tiempo_vam = data.get("TiempoVAM", 0)
            edad = data.get("Edad", 20)
            apache = data.get("APACHE", 0)
            return pd.DataFrame(
                {
                    "Edad": [edad],
                    "Diag.Ing1": [diag_ing1],
                    "Diag.Ing2": [diag_ing2],
                    "Diag.Egr2": [diag_egr2],
                    "TiempoVAM": [tiempo_vam],
                    "APACHE": [apache],
                }
            )

    except Exception as e:
        tb_text = "".join(traceback.format_exception(*sys.exc_info()))
        print(f"Error building prediction data: {e}\n{tb_text}")
        raise


def simulate_and_predict_patient(csv_path: str, selection: int, **kwargs) -> tuple[DataFrame, dict]:
    """
    Run simulation and prediction for a specific patient identified by row index in a CSV.

    Args:
        csv_path: Path to the CSV file containing real data.
        selection: Row index of the patient to process.

    Returns:
        A tuple (simulation_df, prediction_result_dict).
    """

    # Backwards compatibility: accept Spanish kwarg names if provided
    if "ruta_archivo_csv" in kwargs and (csv_path is None or csv_path == ""):
        csv_path = kwargs.get("ruta_archivo_csv")
    if "ruta_fichero_csv" in kwargs and (csv_path is None or csv_path == ""):
        csv_path = kwargs.get("ruta_fichero_csv")
    if "df_selection" in kwargs and (selection is None or selection == 0):
        selection = kwargs.get("df_selection")

    # Extract patient data tuple
    patient_tuple = extract_true_data_from_csv(csv_path, index=selection, return_type="tuple")

    # Run simulation for the patient
    n_runs = kwargs.get("corridas_simulacion", SIM_RUNS_DEFAULT)

    simulation_df = run_experiment(
        n_runs,
        age=int(patient_tuple[0]),
        d1=int(patient_tuple[1]),
        d2=int(patient_tuple[2]),
        d3=int(patient_tuple[3]),
        d4=int(patient_tuple[4]),
        apache=int(patient_tuple[5]),
        resp_insuf=int(patient_tuple[6]),
        artif_vent=int(patient_tuple[7]),
        vam_time=int(patient_tuple[9]),
        uti_stay=int(patient_tuple[8]),
        preuti_stay=int(patient_tuple[10]),
        percent=random.randint(0, 10),
    )

    # Prepare data for prediction
    patient_dict = prepare_patient_data_for_prediction(patient_tuple)

    # Prediction
    try:
        prediction_df = get_data_for_prediction(patient_dict)
        preds, preds_proba = predict(prediction_df)

        prediction_result = {
            "clase_predicha": int(preds[0]),
            "probabilidad_fallecimiento": float(preds_proba[0]),
            "interpretacion": "Fallece" if preds[0] == 1 else "No fallece",
        }
    except Exception as e:
        print(f"Prediction error: {e}")
        prediction_result = {
            "clase_predicha": None,
            "probabilidad_fallecimiento": None,
            "interpretacion": "Prediction error",
        }

    return simulation_df, prediction_result


def prepare_patient_data_for_prediction(patient_tuple: tuple) -> dict:
    """Convert a patient data tuple into a dictionary ready for the prediction model.

    Args:
        patient_tuple: Tuple containing patient data in the following order:
            (edad, diag_ing1, diag_ing2, diag_ing3, diag_ing4, apache, insuf_resp, va,
            est_uci, tiempo_vam, est_preuti, diag_egr2)

    Returns:
        A dict with the keys required by the prediction model (Edad, Diag.Ing1, Diag.Ing2,
        Diag.Egr2, TiempoVAM, APACHE) with values cast to integers.
    """

    return {
        "Edad": int(patient_tuple[0]),
        "Diag.Ing1": int(patient_tuple[1]),
        "Diag.Ing2": int(patient_tuple[2]),
        "Diag.Egr2": int(patient_tuple[11]),
        "TiempoVAM": int(patient_tuple[9]),
        "APACHE": int(patient_tuple[5]),
    }


def apply_theme(theme_name):
    """Apply the selected theme to the Streamlit application."""

    if theme_name == "dark":
        st._config.set_option("theme.base", "dark")
        st._config.set_option("theme.primaryColor", "#66C5A0")
        st._config.set_option("theme.backgroundColor", "#0E1117")
        st._config.set_option("theme.secondaryBackgroundColor", "#262730")
    else:
        st._config.set_option("theme.base", "light")
        st._config.set_option("theme.primaryColor", "#66C5A0")
        st._config.set_option("theme.backgroundColor", "#FFFFF8")
        st._config.set_option("theme.secondaryBackgroundColor", "#F3F6F0")


def simulate_all_true_data(
    true_data: pd.DataFrame | None = None, n_runs: int | None = None, debug: bool = False, seed: int | None = None
) -> np.ndarray:
    """
    Simulate the experiment for every patient present in `true_data` (DataFrame) or in the
    canonical CSV (`FICHERODEDATOS_CSV_PATH`) when `true_data` is None.

    The function leverages the existing `extract_true_data_from_csv` helper to obtain the
    patient-level inputs when possible. It returns a numpy ndarray shaped
    (n_patients, n_runs, n_experiment_variables) where `n_experiment_variables == len(EXP_VARS)`.
    """

    if n_runs is None:
        n_runs = SIM_RUNS_DEFAULT

    # Prefer to use the extractor which already maps CSV rows to the expected dict structure
    if true_data is None:
        records = extract_true_data_from_csv(csv_path=FICHERODEDATOS_CSV_PATH, index=None, as_dataframe=False)
    elif isinstance(true_data, pd.DataFrame):
        # Convert provided DataFrame to the same dict structure used by extract_true_data_from_csv
        df_tmp = true_data.reset_index(drop=True)
        records = []
        for i in range(len(df_tmp)):
            try:
                records.append(build_row_from_dataframe(df_tmp, i))
            except Exception:
                # Skip rows that cannot be parsed
                continue
    else:
        raise ValueError("`true_data` must be a pandas DataFrame or None")

    n_patients = len(records)
    n_vars = len(EXP_VARS)

    sims: list[np.ndarray] = []

    # If seed provided, build a numpy RNG and sample a global percent to match get_true_data_for_validation
    rng = None
    global_percent = None
    if seed is not None:
        rng = np.random.default_rng(seed)
        # replicate get_true_data_for_validation's behavior: integer in [0,10] inclusive
        global_percent = int(rng.integers(low=0, high=10, endpoint=True))

    for rec in records:
        try:
            df_sim = run_experiment(
                n_runs,
                age=int(rec.get("edad", 20)),
                d1=int(rec.get("d1", 0)),
                d2=int(rec.get("d2", 0)),
                d3=int(rec.get("d3", 0)),
                d4=int(rec.get("d4", 0)),
                apache=int(rec.get("apache", 0)),
                resp_insuf=int(rec.get("insuf", 0)),
                artif_vent=int(rec.get("va", 0)),
                vam_time=int(rec.get("tiempo_vam", 0)),
                uti_stay=int(rec.get("estuci", 0)),
                preuti_stay=int(rec.get("estpreuci", 0)),
                percent=(global_percent if global_percent is not None else random.randint(0, 10)),
            )

            # Keep integer hours to avoid scientific notation and preserve discreteness
            arr = df_sim[EXP_VARS].to_numpy(dtype=np.int64)
            if arr.ndim == 1:
                arr = arr.reshape((1, -1))
            sims.append(arr)
        except Exception as e:
            # Log the problematic record so the user can inspect input data
            print(f"Simulation failed for record {rec}. Error: {e}")
            sims.append(np.zeros((n_runs, n_vars), dtype=np.int64))

    try:
        simulation_array = np.stack(sims, axis=0)
    except Exception as e:
        # If stacking fails, log the shapes of individual arrays for debugging
        try:
            shapes = [s.shape for s in sims]
        except Exception:
            shapes = None
        print(f"Error stacking simulation arrays: {e}. Individual shapes: {shapes}")
        simulation_array = np.zeros((n_patients, n_runs, n_vars), dtype=np.int64)

    # Stack into final ndarray
    try:
        simulation_array = np.stack(sims, axis=0)
    except Exception as e:
        # If stacking fails, log the shapes of individual arrays for debugging
        try:
            shapes = [s.shape for s in sims]
        except Exception:
            shapes = None
        print(f"Error stacking simulation arrays: {e}. Individual shapes: {shapes}")
        simulation_array = np.zeros((n_patients, n_runs, n_vars), dtype=np.int64)

    # When debug requested, return additional diagnostic info
    if debug:
        # Compute per-patient means and simple stats
        try:
            sim_means_per_patient = simulation_array.mean(axis=1)
            sim_mean = sim_means_per_patient.mean(axis=0)
            sim_std = sim_means_per_patient.std(axis=0, ddof=1)
        except Exception:
            sim_means_per_patient = None
            sim_mean = None
            sim_std = None

        debug_info = {
            "array": simulation_array,
            "records": records,
            "sim_means_per_patient": sim_means_per_patient,
            "sim_mean": sim_mean,
            "sim_std": sim_std,
        }
        return debug_info

    return simulation_array


def get_true_data_for_validation(seed: int | None = None) -> pd.DataFrame:
    df = pd.read_csv(FICHERODEDATOS_CSV_PATH)

    # Building the dataframe column by column
    # TiempoVAM está en horas; Est. UCI y (probablemente) Est. PostUCI vienen en días → convertir a horas
    col_vam_time: pd.Series[int] = df["TiempoVAM"].round(0).astype(int)
    col_uci_stay_days = df["Est. UCI"].round(0)

    # Asegurar índice consistente si la columna no existe
    if "Est. PostUCI" in df.columns:
        col_uci_post_stay_days = df["Est. PostUCI"].round(0)
    else:
        col_uci_post_stay_days = pd.Series([0] * len(df), index=df.index).round(0)

    col_uci_stay: pd.Series[int] = (col_uci_stay_days * 24).astype(int)
    col_uci_post_stay: pd.Series[int] = (col_uci_post_stay_days * 24).astype(int)

    # Create RNG: if seed is None -> random; otherwise int -> reproducible seed.
    if seed is not None:
        if not isinstance(seed, int) or seed < 0:
            raise ValueError("seed must be a non-negative integer or None")
        rng = np.random.default_rng(seed)
    else:
        rng = np.random.default_rng()

    # percent en [0,10] inclusive
    percent: int = int(rng.integers(low=0, high=10, endpoint=True))

    print(f">>>>> Percentage: {percent} (seed={'fixed' if seed is not None else 'random'})")

    pre_vam: list[int] = []
    post_vam: list[int] = []

    # PreVAM = int((EstadiaUCI - TiempoVAM) * percent)
    # PostVAM = EstadiaUCI - TiempoVAM - PreVAM
    for pos in range(len(df)):
        u = int(col_uci_stay.iloc[pos])  # horas
        t = int(col_vam_time.iloc[pos])  # horas
        diff = max(u - t, 0)
        pre = int(diff * (percent / 100))
        post = diff - pre

        if pos < 5:
            print(f"******* Calculation for {pos}")
            print(f"u: {u}")
            print(f"t: {t}")
            print(f"pre: {pre}")
            print(f"post: {post}")

        pre_vam.append(pre)
        post_vam.append(post)

    # ["Tiempo Pre VAM", "Tiempo VAM", "Tiempo Post VAM", "Estadia UCI", "Estadia Post UCI"]
    values = [pre_vam, col_vam_time.tolist(), post_vam, col_uci_stay.tolist(), col_uci_post_stay.tolist()]

    build_true_data = pd.DataFrame({k: v for k, v in zip(EXP_VARS, values)})

    return build_true_data
