from simpy import Environment
from procesar_datos import *
from datetime import timedelta

class UCI:
    def __init__(self, env:Environment, path:str) -> None:
        self.env = env
        #se crea un evento para un nuevo paciente
        self.nuevo_paciente = env.event()

        self.env.process(self.entrada_paciente_uci(path))
        self.env.process(self.entrada_paciente(path))

    def entrada_paciente(self, path:str):

        #Se obtienen los datos del archivo de entrada y se agregan a la cola de pacientes
        gen_fecha_ing = get_fecha_ingreso(path)

        while True:

            print(f"El paciente llego al hospital a las {self.env.now}h")

            #Se ejecuta el evento nuevo paciente
            self.nuevo_paciente.succeed()
            self.nuevo_paciente = self.env.event()
            #Se calcula la espera y se espera hasta que llegue un nuevo paciente
            fecha_siguiente, fecha = next(gen_fecha_ing)
            espera_ingreso = fecha_siguiente - fecha
            yield self.env.timeout(espera_ingreso.days * 24)

    def entrada_paciente_uci(self, path: str):
        gen_fecha_ing_uci = get_fecha_ing_uci(path)
        gen_fecha_ingreso = get_fecha_ingreso(path)
        gen_estadia_uci = get_estadia_uci(path)
        gen_tiempo_van = get_tiempo_vam(path)
        paciente = 1

        while True:

            yield self.nuevo_paciente

            fecha_ing_uci = next(gen_fecha_ing_uci)
            fecha_ingreso = next(gen_fecha_ingreso)
            espera_uci = fecha_ing_uci - fecha_ingreso
            yield self.env.timeout(espera_uci.days * 24)

            print(f"El paciente {paciente} llega a la uci a las {self.env.now}h")
            print(f"El paciente {paciente} se le pone ventilacion artificial a las {self.env.now}h")
            tiempo_van = next(gen_tiempo_van)

            yield tiempo_van

            print(f"Al paciente {paciente} se le quita la ventilacion a las {self.env.now}h")
            estadia_uci = next(gen_estadia_uci)
            salida_uci = estadia_uci * 24 - tiempo_van

            yield salida_uci

