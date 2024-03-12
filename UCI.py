from simpy import Environment
from procesar_datos import *

class UCI:
    def __init__(self, env:Environment, path:str) -> None:
        self.env = env

        self.env.process(self.entrada_paciente(path))

    def entrada_paciente(self, path:str):

        #Se obtienen los datos del archivo de entrada y se agregan a la cola de pacientes
        gen_fecha_ing = get_fecha_ingreso(path)
        paciente = 0

        while True:
            paciente += 1
            print(f"El paciente {paciente} llego al hospital a las {self.env.now}h")

            #Se calcula la espera y se espera hasta que llegue un nuevo paciente
            try:
                next(gen_fecha_ing)
                fecha_siguiente, fecha = next(gen_fecha_ing)
                espera_ingreso = fecha_siguiente - fecha
                yield self.env.timeout(espera_ingreso.days * 24)
            except StopIteration:
                break
            finally:
                self.env.process(self.entrada_paciente_uci(path, paciente))

    def entrada_paciente_uci(self, path: str, paciente: int):

        #Se inicializan los generadores necesarios
        gen_fecha_ing_uci = get_fecha_ing_uci(path)
        gen_fecha_ingreso = get_fecha_ingreso(path)
        gen_estadia_uci = get_estadia_uci(path)
        gen_tiempo_van = get_tiempo_vam(path)

        while True:

            #Se calcula el tiempo de espera de entrada a la uci y se espera
            fecha_ing_uci = next(gen_fecha_ing_uci)
            fecha_ingreso = next(gen_fecha_ingreso)
            espera_uci = fecha_ing_uci - fecha_ingreso[1]

            yield self.env.timeout(espera_uci.days * 24)

            print(f"El paciente {paciente} llega a la uci a las {self.env.now}h")
            print(f"El paciente {paciente} se le pone ventilacion artificial a las {self.env.now}h")

            #Se espera el tiempo que el paciente pasa en van
            tiempo_van = next(gen_tiempo_van)

            yield self.env.timeout(tiempo_van)

            print(f"Al paciente {paciente} se le quita la ventilacion a las {self.env.now}h")

            #Se calcula la salida de la uci y se espera a que suceda
            estadia_uci = next(gen_estadia_uci)
            salida_uci = estadia_uci * 24 - tiempo_van

            yield self.env.timeout(salida_uci)

            #Se decide a que sala ira el paciente
            gen_sala_egreso = get_sala_egreso(path)
            sala_egreso = next(gen_sala_egreso)
            print(f"El paciente {paciente} salio de la uci y fue trasladado hacia {sala_egreso}")

            #Se espera el egreso del paciente
            gen_fecha_egreso = get_fecha_egreso(path)
            egreso = next(gen_fecha_egreso)

            yield self.env.timeout(egreso.day * 24)

            #Se termina la simulacion
            gen_evolucion = get_evolucion(path)
            evolucion = next(gen_evolucion)

            if evolucion == "vivo":
                print(f"El paciente {paciente} se mantiene vivo y fue dado de alta a las {self.env.now}")
                break
            else:
                print(f"El paciente {paciente} fallece a las {self.env.now}")
                break

env = Environment()
uci = UCI(env, "datos.csv")
env.run()