class Paciente:
    """Entidad que representa un paciente."""

    def __init__(
            self,
            edad=1,
            diag1=0,
            diag2=0,
            diag3=0,
            diag4=0,
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
            diag1: Categoría 1 del diagnóstico del paciente.
            diag2: Categoría 2 del diagnóstico del paciente.
            diag3: Categoría 3 del diagnóstico del paciente.
            diag4: Categoría 4 del diagnóstico del paciente.
            apache: Indicador de Apache del paciente.
            est_uti: Tiempo de estancia en UTI.
            est_pre_uti: Tiempo de estancia Pre-UTI.
            tipo_va: Tipo de ventilación artificial.
            tiempo_va: Tiempo en ventilación artificial.
            porciento_tiempo: Porciento de tiempo en ventilación artificial.
        """

        self._edad = edad
        self._diag1 = diag1
        self._diag2 = diag2
        self._diag3 = diag3
        self._diag4 = diag4
        self._apache = apache
        self._est_uti = est_uti
        self._est_pre_uti = est_pre_uti
        self._tipo_va = tipo_va
        self._tiempo_va = tiempo_va
        self._porciento_tiempo = porciento_tiempo

    @property
    def edad(self):
        return self._edad

    @edad.setter
    def edad(self, edad: int) -> None:
        self._edad = edad

    @property
    def diag1(self):
        return self._diag1

    @diag1.setter
    def diag1(self, diag1: int) -> None:
        self._diag1 = diag1

    @property
    def diag2(self):
        return self._diag2

    @diag2.setter
    def diag2(self, diag2: int) -> None:
        self._diag2 = diag2

    @property
    def diag3(self):
        return self._diag3

    @diag3.setter
    def diag3(self, diag3: int) -> None:
        self._diag3 = diag3

    @property
    def diag4(self):
        return self._diag4

    @diag4.setter
    def diag4(self, diag4: int) -> None:
        self._diag4 = diag4

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

    def to_dict(self) -> dict[str, int | float]:
        """Devuelve los datos del paciente como un diccionario."""

        return {
            "edad": self.edad,
            "apache": self.apache,
            "diag1": self.diag1,
            "diag2": self.diag2,
            "diag3": self.diag3,
            "diag4": self.diag4,
            "est_uti": self.est_uti,
            "est_pre_uti": self.est_pre_uti,
            "tipo_va": self.tipo_va,
            "tiempo_va": self.tiempo_va,
            "porciento_tiempo": self.porciento_tiempo,
        }
