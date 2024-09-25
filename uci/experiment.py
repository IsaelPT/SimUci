import pandas as pd
import simpy

from uci import distribuciones
from uci.simulacion import Simulacion


class Experiment:
    def __init__(self, edad: int, diag_ing1: int, diag_ing2: int, diag_ing3: int, diag_ing4: int,
                 apache: int, insuf_resp: int, va: int, estadia_uti: int, tiempo_vam: int,
                 est_pre_uci: int, porciento: int = 10):
        self.edad = edad
        self.diag1 = diag_ing1
        self.diag2 = diag_ing2
        self.diag3 = diag_ing3
        self.diag4 = diag_ing4
        self.apache = apache
        self.insuf_resp = insuf_resp
        self.va = va
        self.estadia_uti = estadia_uti
        self.tiempo_vam = tiempo_vam
        self.tiempo_pre_uti = est_pre_uci
        self.porciento = porciento

        self.results = {}

    def init_results_variables(self):
        self.results = {"Llegada UCI": 0, "Tiempo Pre VAM": 0, "Comienzo VAM": 0, "Tiempo VAM": 0, "Salida VAM": 0,
                        "Tiempo Post VAM": 0, "Salida UCI": 0, "Estadia Uci": 0, "Estadia Post Uci": 0,
                        "Egreso": 0}


def single_run(experiment):
    env = simpy.Environment()
    experiment.init_results_variables()
    cluster = distribuciones.clustering(experiment.edad, experiment.diag1, experiment.diag2,
                                        experiment.diag3, experiment.diag4, experiment.apache,
                                        experiment.insuf_resp, experiment.va, experiment.estadia_uti,
                                        experiment.tiempo_vam, experiment.tiempo_pre_uti)
    simulacion = Simulacion(experiment, cluster)
    env.process(simulacion.uci(env))
    env.run()

    results = experiment.results
    return results


def multiple_replication(experiment, n_reps=100):
    results = [single_run(experiment) for _ in range(n_reps)]
    df = pd.DataFrame(results)
    return df
