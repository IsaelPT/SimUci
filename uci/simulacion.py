import simpy
import numpy as np
from typing import TYPE_CHECKING

from uci import distribuciones

if TYPE_CHECKING:
    # Import only for type checking to avoid circular runtime import
    from uci.experiment import Experiment


class Simulation:
    def __init__(
        self,
        experiment: "Experiment",
        cluster: np.intp,
    ) -> None:
        self.experiment = experiment
        self.cluster: np.intp = cluster

    def uci(self, env: simpy.Environment):
        # Helper: convert numpy arrays/scalars or lists to a Python float safely
        # This prevents the simulation for spitting zeros due to distribution format issues
        def _to_scalar(x) -> float:
            try:
                # Fast path: plain number
                return float(x)
            except Exception:
                try:
                    # Numpy arrays / scalars
                    arr = np.asarray(x)
                    if arr.size == 0:
                        return 0.0
                    return float(arr.reshape(-1)[0])
                except Exception:
                    return 0.0

        is_cluster_zero: bool = self.cluster == 0

        post_uci = int(
            _to_scalar(
                distribuciones.tiemp_postUCI_cluster0() if is_cluster_zero else distribuciones.tiemp_postUCI_clustet1()
            )
        )
        uci = int(
            _to_scalar(distribuciones.estad_UTI_cluster0() if is_cluster_zero else distribuciones.estad_UTI_cluster1())
        )

        # Ensure VAM does not exceed UCI; cap attempts to avoid infinite loop if distribution heavily skews
        for _ in range(1000):
            vam = int(
                _to_scalar(
                    distribuciones.tiemp_VAM_cluster0() if is_cluster_zero else distribuciones.tiemp_VAM_cluster1()
                )
            )
            if vam <= uci:
                break
        else:
            # As a safety net, clamp VAM to UCI
            vam = uci

        pre_vam = int((uci - vam) * self.experiment.porciento / 100)
        post_vam = uci - pre_vam - vam

        self.experiment.result["Tiempo Post VAM"] = post_vam
        self.experiment.result["Tiempo VAM"] = vam
        self.experiment.result["Tiempo Pre VAM"] = pre_vam
        self.experiment.result["Estadia Post UCI"] = post_uci
        self.experiment.result["Estadia UCI"] = uci

        yield env.timeout(pre_vam)
        yield env.timeout(vam)
        yield env.timeout(post_vam)
        yield env.timeout(post_uci)
