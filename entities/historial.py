import pickle
import traceback
from collections import OrderedDict

from entities.paciente import Paciente
from tools.excepciones import ExceptionHelper
from tools.utils import Rutas


class Historial:
    """Un diccionario historial con los últimos `n` datos de pacientes que se han ingresado con anterioridad."""

    def __init__(self, limite: int = 10):
        """
        Crea un diccionario ordenado con un límite definido a la hora de instanciación.
        Args:
            limite: límite de elementos del historial de pacientes.
        """

        self.ARCHIVO_PATH = Rutas.Data.HISTORIAL_PACIENTES
        self.limite = limite
        self.historial = OrderedDict()
        """Diccionario Ordenado con historial con los últimos `limite` pacientes. Provee la función `popitem()` para eliminar el último elemento."""
        self.cargar_historial()

    def guardar_historial(self) -> None:
        with open(self.ARCHIVO_PATH, "wb") as file:
            pickle.dump(self.historial, file)

    def cargar_historial(self) -> None:
        try:
            with open(self.ARCHIVO_PATH, "rb") as file:
                self.historial = pickle.load(file)
        except Exception as e:
            print(f"Error al cargar el archivo: {self.ARCHIVO_PATH}\n{e}\n{traceback.format_exc()}")
            print("Creando nuevo archivo de datos de Historial...")
            self.guardar_historial()

    def agregar(self, paciente: Paciente):
        """Agrega un paciente al principio del historial. Si se excede el límite de pacientes, se elimina el último, se mueven los pacientes en el diccionario y se ingresa el paciente al principio (key=0)."""

        if len(self.historial) >= self.limite:
            self.historial.popitem(last=True)
            for key in list(self.historial.keys())[::-1]:
                self.historial[key + 1] = self.historial.pop(key)
        self.historial[0] = paciente

    def eliminar(self, key: int = None):
        """Elimina el último paciente o el paciente en la llave especificada."""

        if key is None:
            self.historial.popitem(last=True)
        else:
            self.historial.pop(key)

    def get_paciente(self, paciente: Paciente = None) -> Paciente:
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
            ExceptionHelper().save(e)

    def __str__(self):
        return str(self.historial)
