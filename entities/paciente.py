from tools import utils

DIAGNOSTICOS = utils.VariablesConstantes.DIAG_PREUCI
TIPOS_VA = utils.VariablesConstantes.TIPO_VENT


class Paciente:
    """Entidad que representa un paciente."""

    def __init__(
            self,
            edad=1,
            diag_ing1=1,
            diag_ing2=1,
            diag_ing3=1,
            diag_ing4=1,
            apache=.0,
            est_uti=0,
            est_pre_uti=0,
            tipo_va=0,
            tiempo_va=0,
            porciento_tiempo=.0
    ) -> None:
        """
        Inicia los datos del paciente.

        Aclaración: Los diagnósticos (`diag1`, `diag2`, etc.), tiempo de ventilación artificial `(tipo_va)`, son números
        por motivos de análisis de datos.
        Args:
            edad: Edad del paciente en el rango de 1 a 120 años de edad.
            diag_ing1: Categoría 1 del diagnóstico del paciente.
            diag_ing2: Categoría 2 del diagnóstico del paciente.
            diag_ing3: Categoría 3 del diagnóstico del paciente.
            diag_ing4: Categoría 4 del diagnóstico del paciente.
            apache: Indicador de Apache del paciente.
            est_uti: Tiempo de estancia en UTI.
            est_pre_uti: Tiempo de estancia Pre-UTI.
            tipo_va: Tipo de ventilación artificial.
            tiempo_va: Tiempo en ventilación artificial.
            porciento_tiempo: Porciento de tiempo en ventilación artificial.
        """

        self._edad: int = edad
        self._diag_ing1: int = diag_ing1
        self._diag_ing2: int = diag_ing2
        self._diag_ing3: int = diag_ing3
        self._diag_ing4: int = diag_ing4
        self._apache: float = apache
        self._est_uti: int = est_uti
        self._est_pre_uti: int = est_pre_uti
        self._tipo_va: int = tipo_va
        self._tiempo_va: int = tiempo_va
        self._porciento_tiempo: float = porciento_tiempo

    def to_dict(self, int_str: bool = True) -> dict[str, int | float]:
        """
        Devuelve los datos del paciente como un diccionario.

        Args:
            int_str: Define el formato del diccionario. `True` por defecto (`dict[int: int | float]`).

        Returns:
            Diccionario con estructura definida por parámetro con pares de datos de diagnósticos enumerados y valores.
            Si `True`, diccionario será un set `dict[str: int | float]`, caso contrario, `dict[str: str | int | float]`.
        """
        if int_str:
            return {
                0: self.edad, 1: self.apache, 2: self.diag_ing1, 3: self.diag_ing2, 4: self.diag_ing3, 5: self.diag_ing4,
                6: self.est_uti, 7: self.est_pre_uti, 8: self.tipo_va, 9: self.tiempo_va, 10: self.porciento_tiempo,
            }
        else:
            return {
                "edad": self.edad, "apache": self.apache, "diag1": self.diag_ing1, "diag2": self.diag_ing2, "diag3": self.diag_ing3,
                "diag4": self.diag_ing4, "est_uti": self.est_uti, "est_pre_uti": self.est_pre_uti, "tipo_va": self.tipo_va,
                "tiempo_va": self.tiempo_va, "porciento_tiempo": self.porciento_tiempo,
            }

    @staticmethod
    def translate_diag(diag: str | int) -> int | str:
        """
        Convierte el diagnóstico de un tipo de dato a otro. Si está expresado como `string` y está en la lista
        de diagnósticos, se expresa con su `key` numérico respectivo.
        Args:
            diag: diagnóstico expresado en `str` o `int`.

        Returns:
            `int`- Key del diagnóstico si parámetro es `str` y diagnóstico está contenido en lista de diag.

            `str`- Diagnóstico si parámetro es `int` y este Key está contenido en la lista de diag.
        """
        if isinstance(diag, str):
            for key, value_diagnostico in DIAGNOSTICOS.items():
                if diag == value_diagnostico:
                    return key
            raise Exception(
                f"No se encontró el diagnóstico: <{diag}> -> ({type(diag)}) en el diccionario de diagnósticos.")
        if isinstance(diag, int):
            for key, value_diagnostico in DIAGNOSTICOS.items():
                if diag == key:
                    return value_diagnostico
            raise Exception(f"No se encontró el key: <{diag}> -> ({type(diag)}) en el diccionario de diagnósticos.")
        raise Exception(f"No se logró traducir correctamente el diagnóstico: type(diag) -> {type(diag)}")

    @staticmethod
    def translate_tipo_va(tipo: str | int) -> int | str:
        """
        Convierte el tipo de VA de un tipo de dato a otro. Si está expresado como `string` y está en la lista
        de tipos de VA, se expresa con su `key` numérico respectivo.
        Args:
            tipo: tipo de ventilación expresado en `str` o `int`.

        Returns:
            `int`- Key del tipo de VA si parámetro es `str` y tipo de VA está contenido en lista de tipo de VA.

            `str`- Tipo de VA si parámetro es `int` y este Key está contenido en la lista de tipo de VA.
        """
        if isinstance(tipo, str):
            for key, value_tipo in TIPOS_VA.items():
                if tipo == value_tipo:
                    return key
            raise Exception(f"No se encontró el key: ({tipo} -> {type(tipo)}) en el diccionario de tipos de VA.")
        if isinstance(tipo, int):
            for key, value_tipo in TIPOS_VA.items():
                if tipo == key:
                    return value_tipo
            raise Exception(f"No se encontró el tipo de VA: ({tipo} -> {type(tipo)}) en el diccionario de tipos de VA.")
        raise Exception(f"No se logró traducir correctamente el tipo de VA: type(tipo) -> {type(tipo)}. ")

    @property
    def edad(self):
        return self._edad

    @edad.setter
    def edad(self, edad: int) -> None:
        self._edad = edad

    @property
    def diag_ing1(self):
        return self._diag_ing1

    @diag_ing1.setter
    def diag_ing1(self, diag: int) -> None:
        self._diag_ing1 = diag

    @property
    def diag_ing2(self):
        return self._diag_ing2

    @diag_ing2.setter
    def diag_ing2(self, diag: int) -> None:
        self._diag_ing2 = diag

    @property
    def diag_ing3(self):
        return self._diag_ing3

    @diag_ing3.setter
    def diag_ing3(self, diag: int) -> None:
        self._diag_ing3 = diag

    @property
    def diag_ing4(self):
        return self._diag_ing4

    @diag_ing4.setter
    def diag_ing4(self, diag: int) -> None:
        self._diag_ing4 = diag

    @property
    def apache(self):
        return self._apache

    @apache.setter
    def apache(self, apache: float) -> None:
        self._apache = apache

    @property
    def est_uti(self):
        return self._est_uti

    @est_uti.setter
    def est_uti(self, est_uti: int) -> None:
        self._est_uti = est_uti

    @property
    def est_pre_uti(self):
        return self._est_pre_uti

    @est_pre_uti.setter
    def est_pre_uti(self, est_pre_uti: int) -> None:
        self._est_pre_uti = est_pre_uti

    @property
    def tipo_va(self):
        return self._tipo_va

    @tipo_va.setter
    def tipo_va(self, tipo_va: int) -> None:
        self._tipo_va = tipo_va

    @property
    def tiempo_va(self):
        return self._tiempo_va

    @tiempo_va.setter
    def tiempo_va(self, tiempo_va: int) -> None:
        self._tiempo_va = tiempo_va

    @property
    def porciento_tiempo(self):
        return self._porciento_tiempo

    @porciento_tiempo.setter
    def porciento_tiempo(self, porciento_tiempo: float) -> None:
        self._porciento_tiempo = porciento_tiempo

