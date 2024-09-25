import simpy

from uci import distribuciones


class Simulacion:
    def __init__(self, experiment, cluster) -> None:
        self.experiment = experiment
        self.cluster = cluster

    def uci(self, env: simpy.Environment):

        if self.cluster == 0:
            post_uci = int(distribuciones.tiemp_postUCI0())
            uci = int(distribuciones.estad_UTI0())
            while True:
                vam = int(distribuciones.tiemp_VAM0())
                if vam <= uci:
                    break
        else:
            post_uci = int(distribuciones.tiemp_postUCI1())
            uci = int(distribuciones.estad_UTI1())
            while True:
                vam = int(distribuciones.tiemp_VAM1())
                if vam <= uci:
                    break

        pre_vam = int((uci - vam) * self.experiment.porciento / 100)
        post_vam = uci - pre_vam - vam
        self.experiment.results["Tiempo Post VAM"] = post_vam
        self.experiment.results["Tiempo VAM"] = vam
        self.experiment.results["Tiempo Pre VAM"] = pre_vam
        self.experiment.results["Estadia Post Uci"] = post_uci
        self.experiment.results["Estadia Uci"] = uci

        self.experiment.results["Llegada UCI"] = env.now

        yield env.timeout(pre_vam)
        self.experiment.results["Comienzo VAM"] = env.now

        yield env.timeout(vam)
        self.experiment.results["Salida VAM"] = env.now

        yield env.timeout(post_vam)
        self.experiment.results["Salida UCI"] = env.now

        yield env.timeout(post_uci)
        self.experiment.results["Egreso"] = env.now
