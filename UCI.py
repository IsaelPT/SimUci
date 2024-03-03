from simpy import Environment
from procesar_datos import *


class UCI:
    def __init__(self, env: Environment) -> None:
        self.env = env

    def entrada_paciente(self, path: str):
        fecha_ing = get_fecha_ingreso(path)

        while True:
            print(f"El paciente llego al hospital a las {self.env.now}")
            yield self.env.process(self.entrada_paciente_uci())

            espera_ingreso = -(next(fecha_ing) - next(fecha_ing))
            yield self.env.timeout(espera_ingreso)

    def entrada_paciente_uci(self, path: str):
        fecha_ing_uci = get_fecha_ing_uci(path)

        while True:
            print(f"El paciente llega a la uci a las {self.env.now}")
