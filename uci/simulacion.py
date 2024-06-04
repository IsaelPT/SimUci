import distribuciones
import threading
import simpy
import numpy as np

class Simulacion(threading.Thread):
    def __init__(self, Edad, Diag_Ing1, Diag_Ing2, Diag_Ing3, Diag_Ing4, APACHE, InsufResp, VA, EstadiaUTI, TiempoVAM, Est_PreUCI, porciento, cant_corr) -> None:
        super().__init__()

        self.edad = Edad
        self.diag1 = Diag_Ing1
        self.diag2 = Diag_Ing2
        self.diag3 = Diag_Ing3
        self.diag4 = Diag_Ing4
        self.apache = APACHE
        self.insuf_resp = InsufResp
        self.va = VA
        self.estadia_uti = EstadiaUTI
        self.tiempo_vam = TiempoVAM
        self.tiemp_pre_uti = Est_PreUCI
        self.porciento = porciento
        self.cant_corr = cant_corr

        self.tiemp_post_vam = []
        self.tiemp_vam = []
        self.tiemp_pre_vam = []
        self.tiemp_post_uci = []
        self.tiemp_uci = []

        self._stop_event = threading.Event()

    def run(self):
        env = simpy.Environment()

        for _ in range(self.cant_corr):
            env.process(self.uci(env))

        env.run()
        self.cal_estadisticas()

    def stop(self):
        self._stop_event.set()

    def uci(self, env:simpy.Environment):
        cluster = distribuciones.clustering(self.edad, self.diag1, self.diag2, self.diag3,
                                            self.diag4, self.apache, self.insuf_resp, self.va,
                                            self.estadia_uti, self.tiempo_vam, self.tiemp_pre_uti)

        vam = 0
        post_uci = 0
        uci = 0
        if cluster == 0:
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

        pre_vam = int((uci - vam) * self.porciento / 100)
        post_vam = uci - pre_vam - vam

        self.tiemp_post_vam.append(post_vam)
        self.tiemp_vam.append(vam)
        self.tiemp_pre_vam.append(pre_vam)
        self.tiemp_post_uci.append(post_uci)
        self.tiemp_uci.append(uci)

        yield env.timeout(pre_vam)
        yield env.timeout(vam)
        yield env.timeout(post_vam)
        yield env.timeout(post_uci)

    def cal_estadisticas(self):
        mean_post_vam = np.mean(self.tiemp_post_vam)
        mean_vam = np.mean(self.tiemp_vam)
        mean_pre_vam = np.mean(self.tiemp_pre_vam)
        mean_post_uci = np.mean(self.tiemp_post_uci)
        mean_uci = np.mean(self.tiemp_uci)
        print(f"Las medias de las varibles son:\nTiempo post VAM: {mean_post_vam}\nTiempo en VAM: {mean_vam}\nTiempo antes del VAM: {mean_pre_vam}\nTiempo en UCI: {mean_uci}\nTiempo luego de salir de la UCI: {mean_post_uci}")

sim = Simulacion(35,1,0,0,0,10,8,1,96,48,0,10,1000)
sim.run()