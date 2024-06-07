import json
import traceback
from typing import TYPE_CHECKING

from PyQt5.QtCore import Qt
from PyQt5.QtGui import QIcon
from PyQt5.QtWidgets import QWidget, QTableWidgetItem
from PyQt5.uic import loadUi

from entities.paciente import Paciente
from tools.excepciones import ExceptionSaver
from tools.utils import Rutas, VariablesConstantes

if TYPE_CHECKING:
    from ui_code.mainmenu_win import MainMenuWindow


class SimulationWindow(QWidget):
    main_menu_win: 'MainMenuWindow' = None
    dict_tiposVA: dict[int, str]
    dict_diag_preuci: dict[int, str]
    paciente: Paciente = None
    corridas_simulacion: int

    def __init__(self, parent) -> None:
        super().__init__()
        self.main_menu_win = parent  # Referencia a la clase padre (MainMenuWindow).
        self.dict_diag_preuci = {}  # Diccionario con todas las categorías de diagnósticos de ingreso.
        self.dict_tiposVA = {}  # Tipos de ventilación artificial.
        self.paciente = Paciente()
        self.historial = {}  # Diccionario que sostiene el historial con los últimos 10 pacientes.

        loadUi(Rutas.UiFiles.SIMULATIONWIDGET_UI, self)  # baseinstance: SimulationWindow

        self._init_fields()
        self._init_components()
        self._connect_signals()

    def _init_fields(self) -> None:
        self.dict_diag_preuci = VariablesConstantes.DIAG_PREUCI
        self.dict_tiposVA = VariablesConstantes.TIPO_VENT

        # Obtener todos los valores presentes en la GUI
        self.paciente.edad = self.sb_edad.value()
        self.paciente.apache = self.sb_apache.value()
        self.paciente.est_uti = self.sb_estadiaUTI.value()
        self.paciente.est_pre_uti = self.sb_estadiaPreUTI.value()
        self.paciente.tiempo_va = self.sb_tiempoVA.value()
        self.paciente.porciento_tiempo = self.sb_porciento_tiempo.value()
        self.paciente.diag1 = self.cb_d1.currentText()
        self.paciente.diag2 = self.cb_d2.currentText()
        self.paciente.diag3 = self.cb_d3.currentText()
        self.paciente.diag4 = self.cb_d4.currentText()
        self.paciente.tiempo_va = self.cb_tipoVA.currentText()

    def _init_components(self) -> None:
        self.setWindowIcon(QIcon(Rutas.Iconos.WINDOWICON_SIMULATION))
        self.pb_detener.setEnabled(False)

        # Configurando los ComboBox
        list_diag_preuci = list(self.dict_diag_preuci.values())
        list_tiposVA = list(self.dict_tiposVA.values())
        self.cb_d1.addItems(list_diag_preuci)
        self.cb_d2.addItems(list_diag_preuci)
        self.cb_d3.addItems(list_diag_preuci)
        self.cb_d4.addItems(list_diag_preuci)
        self.cb_tipoVA.addItems(list_tiposVA)

    def _connect_signals(self) -> None:
        # PushButtons
        self.pb_comenzar.clicked.connect(self.comenzar_simulacion)
        self.pb_detener.clicked.connect(self.detener_simulacion)
        self.pb_salir.clicked.connect(self.cerrar_ventana)

        # Toolboxes
        self.tb_deshacer.clicked.connect(self.deshacer_eliminacion_historial)
        self.tb_eliminar.clicked.connect(self.eliminar_paciente_historial)
        self.tb_limpiar.clicked.connect(self.limpiar_historial)

        # SpinBoxes
        self.sb_edad.valueChanged.connect(self.update_edad)
        self.sb_apache.valueChanged.connect(self.update_apache)
        self.sb_estadiaUTI.valueChanged.connect(self.update_est_uci)
        self.sb_estadiaPreUTI.valueChanged.connect(self.update_est_pre_uci)
        self.sb_tiempoVA.valueChanged.connect(self.update_tiempo_va)
        self.sb_porciento_tiempo.valueChanged.connect(self.update_porciento_tiempo)
        self.sb_corridas_simulacion.valueChanged.connect(self.update_corridas_sim)

        # ComboBoxes
        self.cb_d1.currentIndexChanged[str].connect(lambda diag1: self.update_diag(diag1, 1))
        self.cb_d2.currentIndexChanged[str].connect(lambda diag2: self.update_diag(diag2, 2))
        self.cb_d3.currentIndexChanged[str].connect(lambda diag3: self.update_diag(diag3, 3))
        self.cb_d4.currentIndexChanged[str].connect(lambda diag4: self.update_diag(diag4, 4))
        self.cb_tipoVA.currentIndexChanged[str].connect(lambda tipo: self.update_tipo_va(tipo))

    def comenzar_simulacion(self) -> None:
        try:
            print("Clickeado botón COMENZAR")
            self._update_historial()
        except Exception as e:
            print(f"Ha ocurrido un error al COMENZAR la simulación: {e}\n{traceback.format_exc()}")
            ExceptionSaver().save(e)

    def detener_simulacion(self) -> None:
        try:
            print("Clickeado botón DETENER")
        except Exception as e:
            print(f"Ha ocurrido un error al DETENER la simulación: {e}\n{traceback.format_exc()}")
            ExceptionSaver().save(e)

    def cerrar_ventana(self) -> None:
        print("Clickeado botón CERRAR")
        try:
            self.main_menu_win.cerrar_ventana_simulacion()
        except Exception as e:
            print(f"{e}:\n{traceback.format_exc()}")

    def eliminar_paciente_historial(self) -> None:
        """Elimina un paciente de la tabla de historial de pacientes."""

        current_row = self.tableWidgetPacientes.currentRow()
        if current_row >= 0:
            self.tableWidgetPacientes.removeRow(current_row)

    def limpiar_historial(self) -> None:
        pass

    def deshacer_eliminacion_historial(self):
        """Deshace las últimas 5 eliminaciones realizadas en el historial de pacientes."""
        pass

    def gestionar(self) -> None:
        print(
            f"Datos del paciente:\n"
            f"Edad: {self.paciente.edad}\n"
            f"Apache: {self.paciente.apache}\n"
            f"Diagnostico1: {self.paciente.diag1}\n"
            f"Diagnostico2: {self.paciente.diag2}\n"
            f"Diagnostico3: {self.paciente.diag3}\n"
            f"Diagnostico4: {self.paciente.diag4}\n"
            f"Tipo VA: {self.paciente.tipo_va}\n"
            f"Tiempo VA: {self.paciente.tiempo_va}\n"
            f"Porciento Tiempo: {self.paciente.porciento_tiempo}\n"
            f"Estancia UTI: {self.paciente.est_uti}\n"
            f"Estancia PreUTI: {self.paciente.est_pre_uti}\n"
        )

    def update_edad(self, edad: int):
        try:
            self.paciente.edad = edad
            print(f"Edad del paciente actualizada a: {edad}")
        except ValueError as e:
            print(e)
            ExceptionSaver().save(e)

    def update_apache(self, apache: float):
        try:
            self.paciente.apache = apache
            print(f"Apache del paciente actualizado a: {apache}")
        except ValueError as e:
            print(e)
            ExceptionSaver().save(e)

    def update_est_uci(self, est_uci: int):
        try:
            self.paciente.est_uti = est_uci
            print(f"Tiempo de estadía UTI del paciente actualizado a: {est_uci}")
        except ValueError as e:
            print(e)
            ExceptionSaver().save(e)

    def update_est_pre_uci(self, est_pre_uci: int):
        try:
            self.paciente.est_pre_uti = est_pre_uci
            print(f"Tiempo de estadía pre-UTI del paciente actualizado a: {est_pre_uci}")
        except ValueError as e:
            print(e)
            ExceptionSaver().save(e)

    def update_tiempo_va(self, tiempo_va: int):
        try:
            self.paciente.tiempo_va = tiempo_va
            print(f"Tiemo de ventilación artificial del paciente actualizado a: {tiempo_va}")
        except ValueError as e:
            print(e)
            ExceptionSaver().save(e)

    def update_porciento_tiempo(self, porciento: float):
        try:
            self.paciente.porciento_tiempo = porciento
            print(f"Apache del paciente actualizado a: {porciento}")
        except ValueError as e:
            print(e)
            ExceptionSaver().save(e)

    def update_corridas_sim(self, corridas_sim: int):
        try:
            self.corridas_simulacion = corridas_sim
            print(f"Número de corridas de la simulación actualizadas a: {corridas_sim}")
        except ValueError as e:
            print(e)

    def update_diag(self, diag: str, tipo_diag: int):
        try:
            int_diag = self._translate_diag(diag)
            if tipo_diag == 1:
                self.paciente.diag1 = int_diag
            elif tipo_diag == 2:
                self.paciente.diag2 = int_diag
            elif tipo_diag == 3:
                self.paciente.diag3 = int_diag
            elif tipo_diag == 4:
                self.paciente.diag4 = int_diag
            else:
                raise Exception("Tipo de diagnóstico no válido (1, 2, 3 o 4 solo posibles).")
            print(f"Diagnóstico {tipo_diag} del paciente actualizado a: {diag} ({int_diag})")
        except ValueError as e:
            print(e)

    def update_tipo_va(self, str_tipoVA: str):
        try:
            int_tipoVA = self._translate_tipo_va(str_tipoVA)
            print(f"Ventilación artificial {int_tipoVA} ({str_tipoVA}) actualizado a: {str_tipoVA} ({int_tipoVA})")
        except ValueError as e:
            print(e)
            ExceptionSaver().save(e)

    def _translate_diag(self, diag: str | int) -> int | str:
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
            for key, value_diagnostico in self.dict_diag_preuci.items():
                if diag == value_diagnostico:
                    return key
            raise Exception(
                f"No se encontró el diagnóstico: {diag} -> ({type(diag)}) en el diccionario de diagnósticos.")
        if isinstance(diag, int):
            for key, value_diagnostico in self.dict_diag_preuci.items():
                if diag == key:
                    return value_diagnostico
            raise Exception(f"No se encontró el key: {diag} -> ({type(diag)}) en el diccionario de diagnósticos.")
        raise Exception(f"No se logró traducir correctamente el diagnóstico: type(diag) -> {type(diag)}")

    def _translate_tipo_va(self, tipo: str | int) -> int | str:
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
            for key, value_tipo in self.dict_tiposVA.items():
                if tipo == value_tipo:
                    return key
            raise Exception(f"No se encontró el key: ({tipo} -> {type(tipo)}) en el diccionario de tipos de VA.")
        if isinstance(tipo, int):
            for key, value_tipo in self.dict_tiposVA.items():
                if tipo == key:
                    return value_tipo
            raise Exception(f"No se encontró el tipo de VA: ({tipo} -> {type(tipo)}) en el diccionario de tipos de VA.")
        raise Exception(f"No se logró traducir correctamente el tipo de VA: type(tipo) -> {type(tipo)}. ")

    def _update_historial(self) -> None:
        try:
            if self.paciente:
                row = self.tableWidgetPacientes.rowCount()  # Última fila de la tabla del historial.
                self.tableWidgetPacientes.insertRow(row)  # Agregar una nueva  fila.

                # Datos del Paciente.
                data = [
                    self.paciente.edad,
                    self.paciente.apache,
                    self._translate_diag(self.paciente.diag1),
                    self._translate_diag(self.paciente.diag2),
                    self._translate_diag(self.paciente.diag3),
                    self._translate_diag(self.paciente.diag4),
                    self.paciente.est_uti,
                    self.paciente.est_pre_uti,
                    self._translate_tipo_va(self.paciente.tipo_va),
                    self.paciente.tiempo_va,
                    self.paciente.porciento_tiempo,
                ]

                # Datos del Paciente parseados a str
                data_to_str = map(lambda d: str(d), data)

                # Items
                items = [QTableWidgetItem(d) for d in data_to_str]
                for _ in items:  # << Alinear los items al centro >>
                    _.setTextAlignment(Qt.AlignCenter)

                # Asignación → Actualizar Tabla.
                try:
                    for columna, item in enumerate(items):
                        self.tableWidgetPacientes.setItem(row, columna, item)
                except Exception as e:
                    print(e)
                self.tableWidgetPacientes.resizeColumnsToContents()
            else:
                raise Exception("Paciente no encontrado.")
        except Exception as e:
            print(e)
            ExceptionSaver().save(e)

    def _update_tabla_historial(self) -> None:
        pass

    def _update_fields(self) -> None:
        pass

    def _guardar_historial(self) -> None:
        with open(Rutas.Data.DATOS_HISTORIAL_PACIENTES, 'a') as json_historial:
            json.dump(self.historial, json_historial)

    def _cargar_historial(self) -> None:
        with open(Rutas.Data.DATOS_HISTORIAL_PACIENTES, 'r') as json_historial:
            datos = json.load(json_historial)
