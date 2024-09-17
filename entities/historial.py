import json
from collections import OrderedDict

from entities.paciente import Paciente
from tools.excepciones import ExceptionSaver
from tools.utils import Rutas


class Historial:
    """Un diccionario historial con los últimos `n` datos de pacientes que se han ingresado con anterioridad."""

    def __init__(self, limite: int=10):
        """
        Crea un diccionario ordenado con un límite definido a la hora de instanciación.
        Args:
            limite: límite de elementos del historial de pacientes.
        """

        self.ARCHIVO_PATH = Rutas.Data.HISTORIAL_PACIENTES
        self.limite = limite
        self.historial = OrderedDict()
        """Diccionario Ordenado con historial con los últimos `limite` pacientes.
        Provee la función `popitem()` para eliminar el último elemento."""
        self.cargar_historial()

    def agregar(self, paciente: Paciente):
        """Agrega un paciente al principio del historial. Si se excede el límite de pacientes, se elimina el último, se
        mueven los pacientes en el diccionario y se ingresa el paciente al principio (key=0)."""

        if len(self.historial) >= self.limite:
            self.historial.popitem(last=True)
            for key in list(self.historial.keys())[::-1]:
                self.historial[key + 1] = self.historial.pop(key)
        self.historial[0] = paciente

    def eliminar(self, key: int=None):
        """Elimina el último paciente o el paciente en la llave especificada."""

        if key is None:
            self.historial.popitem(last=True)
        else:
            self.historial.pop(key)

    def get_paciente(self, paciente: Paciente=None):
        """Obtiene el primer paciente o el paciente especificado del historial de pacientes."""

        try:
            if paciente is None:
                return self.historial[0]
            else:
                for v in self.historial.values():
                    if v == paciente:
                        return paciente
        except Exception as e:
            print(e)
            ExceptionSaver().save(e)

    def guardar_historial(self) -> None:
        """Guarda los datos del historial en un archivo `JSON`."""
        historial_serializable = {key: paciente.to_dict(False) for key, paciente in self.historial.values()}
        print(f">>> Historial Serializable: {historial_serializable}")

        with open(self.ARCHIVO_PATH, "w", encoding="UTF-8") as archivo_json:
            # print(str(json.dumps(self.historial)))
            json.dump(historial_serializable, archivo_json, indent=4)

    def cargar_historial(self) -> None:
        """Carga los datos del historial del archivo serializado."""

        try:
            with open(self.ARCHIVO_PATH, 'r') as archivo_json:
                self.historial = json.load(archivo_json)
        except Exception as e:
            print(e)
            ExceptionSaver().save(e)
            print(f"No se encontró un archivo JSON: {self.ARCHIVO_PATH}")

            # En caso de que no exista un archivo JSON, se crea uno nuevo.
            try:
                print("Creando archivo JSON...")
                self.agregar(Paciente())
                self.guardar_historial()
                self.cargar_historial()
            except Exception as e:
                print(e)
                ExceptionSaver().save(e)

    def __str__(self):
        return str(self.historial)
