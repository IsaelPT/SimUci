import pandas as pd
import simpy

from utils.constants import EXPERIMENT_VARIABLES as VARIABLES_EXPERIMENTO
from uci import distribuciones
from uci.simulacion import Simulacion


class Experiment:
    def __init__(
        self,
        edad: int,
        diagnostico_ingreso1: int,
        diagnostico_ingreso2: int,
        diagnostico_ingreso3: int,
        diagnostico_ingreso4: int,
        apache: int,
        insuficiencia_respiratoria: int,
        ventilacion_artificial: int,
        estadia_uti: int,
        tiempo_vam: int,
        tiempo_estadia_pre_uti: int,
        porciento: int = 10,
    ):
        self.edad = edad
        self.diagn1 = diagnostico_ingreso1
        self.diagn2 = diagnostico_ingreso2
        self.diagn3 = diagnostico_ingreso3
        self.diagn4 = diagnostico_ingreso4
        self.apache = apache
        self.insuf_resp = insuficiencia_respiratoria
        self.va = ventilacion_artificial
        self.estadia_uti = estadia_uti
        self.tiempo_vam = tiempo_vam
        self.tiempo_pre_uti = tiempo_estadia_pre_uti
        self.porciento = porciento

        self.result = {}

    def init_results_variables(self) -> None:
        self.result = {valor: 0 for valor in VARIABLES_EXPERIMENTO}
        # self.result = {"Tiempo Pre VAM": 0, "Tiempo VAM": 0, "Tiempo Post VAM": 0,
        #                "Estadia UCI": 0, "Estadia Post UCI": 0}


def single_run(experiment) -> dict[str, int]:
    env = simpy.Environment()
    experiment.init_results_variables()
    cluster = distribuciones.clustering(
        experiment.edad,
        experiment.diagn1,
        experiment.diagn2,
        experiment.diagn3,
        experiment.diagn4,
        experiment.apache,
        experiment.insuf_resp,
        experiment.va,
        experiment.estadia_uti,
        experiment.tiempo_vam,
        experiment.tiempo_pre_uti,
    )
    simulacion = Simulacion(experiment, cluster)
    env.process(simulacion.uci(env))
    env.run()

    result = experiment.result
    return result


def multiple_replication(experiment: Experiment, n_reps: int = 100) -> pd.DataFrame:
    # Inicializar una lista para almacenar los resultados
    results = []

    # Ejecutar las réplicas de la simulación
    for _ in range(n_reps):
        # Obtener el resultado de una ejecución
        result = single_run(experiment)
        # Asegurarse de que todos los valores sean numéricos
        numeric_result = {}
        for key, value in result.items():
            try:
                # Intentar convertir a entero
                numeric_result[key] = int(float(value))
            except (ValueError, TypeError):
                # Si falla, usar 0 como valor por defecto
                numeric_result[key] = 0
        results.append(numeric_result)

    # Crear el DataFrame con los resultados ya convertidos
    df = pd.DataFrame(results)

    # Verificar que no haya valores nulos
    if df.isnull().any().any():
        df = df.fillna(0)

    # Asegurar que todas las columnas sean enteras
    for col in df.columns:
        df[col] = df[col].astype("int64")

    return df
