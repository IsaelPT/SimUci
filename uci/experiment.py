import pandas as pd
import simpy

from utils.constants import EXPERIMENT_VARIABLES as VARIABLES_EXPERIMENTO
from uci import distribuciones
from uci.simulacion import Simulation


class Experiment:
    def __init__(
        self,
        age: int,
        diagnosis_admission1: int,
        diagnosis_admission2: int,
        diagnosis_admission3: int,
        diagnosis_admission4: int,
        apache: int,
        respiratory_insufficiency: int,
        artificial_ventilation: int,
        uti_stay: int,
        vam_time: int,
        preuti_stay_time: int,
        percent: int = 10,
    ):
        self.edad = age
        self.diagn1 = diagnosis_admission1
        self.diagn2 = diagnosis_admission2
        self.diagn3 = diagnosis_admission3
        self.diagn4 = diagnosis_admission4
        self.apache = apache
        self.insuf_resp = respiratory_insufficiency
        self.va = artificial_ventilation
        self.estadia_uti = uti_stay
        self.tiempo_vam = vam_time
        self.tiempo_pre_uti = preuti_stay_time
        self.porciento = percent

        self.result = {}

    def init_results_variables(self) -> None:
        self.result = {valor: 0 for valor in VARIABLES_EXPERIMENTO}
        # self.result = {"Tiempo Pre VAM": 0, "Tiempo VAM": 0, "Tiempo Post VAM": 0, "Estadia UCI": 0, "Estadia Post UCI": 0}


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
    simulation = Simulation(experiment, cluster)
    env.process(simulation.uci(env))
    env.run()

    result = experiment.result
    return result


def multiple_replication(experiment: Experiment, n_reps: int = 100) -> pd.DataFrame:
    results = []

    for _ in range(n_reps):
        result = single_run(experiment)

        # Asegurarse de que todos los valores sean numéricos
        numeric_result = {}
        for key, value in result.items():
            try:
                numeric_result[key] = int(float(value))
            except (ValueError, TypeError):
                numeric_result[key] = 0
        results.append(numeric_result)

    df = pd.DataFrame(results)

    # Verificar que no haya valores nulos
    if df.isnull().any().any():
        df = df.fillna(0)

    # Asegurar que todas las columnas sean enteras
    for col in df.columns:
        df[col] = df[col].astype("int64")

    return df
