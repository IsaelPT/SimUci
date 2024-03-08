from simpy import Environment
from procesar_datos import *
from datetime import timedelta

class UCI():
    def __init__(self, env:Environment, path:str) -> None:
        self.env = env
        #se crea un evento para un nuevo paciente
        self.nuevo_paciente = env.event()

        self.env.process(self.entrada_paciente_uci(path))
        self.env.process(self.entrada_paciente(path))

    def entrada_paciente(self, path:str):

        #Se obtienen los datos del archivo de entrada y se agregan a la cola de pacientes
        fecha_ing = get_fecha_ingreso(path)

        while True:

            print(f"El paciente llego al hospital a las {self.env.now}h")

            #Se ejecuta el evento nuevo paciente
            self.nuevo_paciente.succeed()
            self.nuevo_paciente = self.env.event()
            #Se calcula la espera y se espera hasta que entre el paciente a la UCI
            fecha_siguiente, fecha = next(fecha_ing)
            espera_ingreso = fecha_siguiente - fecha
            yield self.env.timeout(espera_ingreso.days * 24)

    def entrada_paciente_uci(self, path:str):

        fecha_ing_uci = get_fecha_ing_uci(path)
        fecha_ingreso = get_fecha_ingreso(path)

        while True:

            yield self.nuevo_paciente

            espera_uci = next(fecha_ing_uci) - next(fecha_ingreso)
            yield self.env.timeout(espera_uci.days * 24)

            print(f"El paciente llega a la uci a las {self.env.now}h")

