import time
import traceback
from typing import List
import pandas as pd

from PyQt5 import QtCore
from PyQt5.QtCore import QObject, QThread
from PyQt5.QtWidgets import QWidget, QFileDialog, QMessageBox
from PyQt5.QtGui import QStandardItem, QStandardItemModel
from PyQt5.uic import loadUi
from simpy import Environment

from qt_py.constantes import Rutas, Estilos
from uci import procesar_datos as proc_d

from uci.uci_simulacion import Uci


class SimulationWindow(QWidget):
    """
    Es una ventana donde se llevan a cabo simulaciones con datos brindados a través
    de un `.csv` y se muestran por pantalla los datos.

    Responsabilidades
    -----------------

    - `cargar_csv(self)`: Método que permite cargar archivos `.csv`.
    - `comenzar_simulacion(self)`: Método que inicia el proceso de simulación.
    - `detener_simulacion(self)`: Método para detener la simulación en curso.
    - `cerrar_ventana(self)`: Método para cerrar la ventana de simulación.
    - `_init_tabla_diagnosticos(self, diagnosticos):` Inicializa la tabla de diagnósticos.
    - `_upgrade_progressBarr(self, contador)`: Actualiza el progreso de la barra que visualiza el proceso de la simulación.
    """

    ruta_archivo_csv: str = None

    def __init__(self, main_win) -> None:
        super().__init__()
        self.main_win = main_win
        loadUi(Rutas.SIMULATIONWIDGET_UI, self)  # baseinstance: SimulationWindow

        # QLineEdit para visualizar la dirección donde se localiza el archivo.
        self.lineEdit_ruta_datos.setText("Ruta de archivo...")

        # QColumnView para visualizar los diagnosticos / porcientos.
        self.modelo_tabla = QStandardItemModel(self)
        self.modelo_tabla.setHorizontalHeaderItem(0, QStandardItem("Diagnosticos"))
        self.modelo_tabla.setHorizontalHeaderItem(1, QStandardItem("Porcientos (%)"))
        self.tableView_diagnosticos.setModel(self.modelo_tabla)
        self.tableView_diagnosticos.resizeColumnsToContents()

        self.thread = {}  # Hilos de esta ventana.

        # Conexiones de los componentes.
        self.pB_cargar.clicked.connect(self.cargar_csv)
        self.pB_comenzar.clicked.connect(self.comenzar_simulacion)
        self.pB_detener.clicked.connect(self.detener_simulacion)
        self.pB_salir.clicked.connect(self.cerrar_ventana)

        # Estilos personalizados a los componentes.
        self.pB_cargar.setStyleSheet(Estilos.botones["botones_acciones_verdes"])
        self.pB_comenzar.setStyleSheet(Estilos.botones["botones_acciones_verdes"])
        self.pB_detener.setStyleSheet(Estilos.botones["botones_acciones_verdes"])
        self.pB_salir.setStyleSheet(Estilos.botones["botones_acciones_verdes"])

        # Ajustes iniciales a botones.
        self.pB_detener.setEnabled(False)

    def cargar_csv(self) -> None:
        """
        Carga el archivo .csv a través de un `QFileDialog`.
        """

        self.ruta_archivo_csv, _ = QFileDialog.getOpenFileName(
            self, "Abrir CSV", "", "Archivos CSV (*.csv)"
        )

        if self.ruta_archivo_csv is not None:
            try:
                self.datos_csv = pd.read_csv(self.ruta_archivo_csv)
                diagnosticos = proc_d.get_diagnostico_list(self.ruta_archivo_csv)
                self._init_tabla_diagnosticos(diagnosticos)  # Ingresar datos en Tabla.
                self.lineEdit_ruta_datos.setText(self.ruta_archivo_csv)
                print(f"Archivo CSV '{self.ruta_archivo_csv}' cargado correctamente!")
            except:
                print(
                    f"Ocurrió un error al cargar el archivo:\n{traceback.format_exc()}"
                )
        else:
            raise Exception("La ruta de archivos no existe u ocurrió un error.")

    def comenzar_simulacion(self) -> None:
        """
        Da inicio a la simulación al ser pulsado el botón "Comenzar Simulación".
        """

        if self.ruta_archivo_csv is None:
            QMessageBox.warning(
                self,
                "Imposible iniciar simulación",
                "No se puede iniciar la simulación debido a que no hay datos para simular. Por favor, cargue los datos.",
            )
            return
        try:
            print("-- Se presionó el botón de 'Comenzar Simulacion' --")
            self.thread[1] = Simulation_Thread(
                self,
                self.ruta_archivo_csv,
                proc_d.get_diagnostico_list(self.ruta_archivo_csv),
                self._get_porcientos_de_tabla(),
            )
            if self.ruta_archivo_csv is not None:
                self.thread[1].start()
            else:
                raise Exception("La ruta del archivo .csv está vacía.")
            self.pB_comenzar.setEnabled(False)
            self.pB_detener.setEnabled(True)
            self.pB_cargar.setEnabled(False)
            self.thread[1].signal_progBarr.connect(self._update_progressBarr)
            self.thread[1].signal_terminated.connect(self.pB_comenzar.setEnabled)
            self.thread[1].signal_terminated.connect(self.pB_detener.setEnabled)
        except:
            print(
                f"Ocurrió un error inesperado a la hora de correr la simulación:\n{traceback.format_exc()}"
            )

    def detener_simulacion(self, show_warning_message: bool) -> None:
        """
        Detiene la simulación a medio proceso.

        Parámetros
        ----------
        show_warning_message : bool
            Indica si se debe mostrar un mensaje de advertencia.
        """

        try:
            print("Se presionó el botón de 'Detener Simulación'.")
            self.thread[1].stop()
            self.progressBar.setValue(0)
            self.pB_cargar.setEnabled(True)
            self.pB_comenzar.setEnabled(True)
            self.pB_detener.setEnabled(False)
            if not show_warning_message:
                QMessageBox().warning(
                    self, "Detención de simulación", "Se ha detenido la simulación."
                )
        except:
            print(f"Ocurrió un error inesperado:\n{traceback.format_exc()}")

    def cerrar_ventana(self) -> None:
        """
        Cierra esta ventana de Simulación.
        """

        try:
            self.detener_simulacion(True)
            self.close()
            self.main_win.show()
        except:
            print(f"Ocurrió un error al cerrar la ventana:\n{traceback.format_exc()}")

    def _init_tabla_diagnosticos(self, diagnosticos) -> None:
        """
        Define el modelo del `QColumnView` para visualizar los diagnosticos y porcientos
        que se obtienen de los pacientes.

        Parámetros
        ----------

        diagnosticos
            Una lista que contiene todos los diagnosticos que se obtuvieron del archivo `.csv`.
        """

        self.FILAS = len(diagnosticos)
        print(f"Cantidad de diagnosticos: {self.FILAS}")
        print(f"Lista de diagnosticos importada:\n{diagnosticos}")

        for d in diagnosticos:
            item_diagnostico = QStandardItem(d)
            item_porciento = QStandardItem("0")
            self.modelo_tabla.appendRow([item_diagnostico, item_porciento])

        self.tableView_diagnosticos.resizeColumnsToContents()

    def _update_progressBarr(self, contador: int) -> None:
        """
        Actualiza el progreso de la barra de simulación.

        Parámetros
        ----------

        contador : int
            El número a colocar en la barra del contador.
        """

        self.progressBar.setValue(contador)

    def _get_porcientos_de_tabla(self) -> List[float]:
        """
        Itera sobre los elementos de la segunda columna de la tabla
        (la segunda columna contiene los porcientos) y los guarda en una lista de valores flotantes
        entre 0 y 100.

        Retorna
        -------

        List[float]
            Una lista que contiene los porcentajes de la tabla.
        """

        porcentajes = []
        incorrectos = []
        for index in range(self.modelo_tabla.rowCount()):
            item_porcentaje = self.modelo_tabla.item(index, 1)
            porcentaje: str = item_porcentaje.text()
            if porcentaje.isdigit():
                porcentajes.append(float(item_porcentaje.text()))
            else:
                incorrectos.append(index + 1)
        if len(incorrectos) == 1:
            title = "Porciento incorrecto"
            msg = f"Se ha encontrado que en la columna de porcentajes, precisamente en la fila {incorrectos[0]}, un porciento ha sido ingresado incorrectamente. Por favor, rectifique para poder iniciar la simulación."
            QMessageBox.warning(self, title, msg)
        if len(incorrectos) > 1:
            title = "Porcientos incorrectos"
            msg = f"Se han encontrado que en la columna de porcentajes, precisamente en las filas {incorrectos}, porcientos han sido ingresado incorrectamente. Por favor, rectifique para poder iniciar la simulación."
            QMessageBox.warning(self, title, msg)
        print(f"Lista de porcentajes:\n{porcentajes}")
        return porcentajes


class Simulation_Thread(QThread):
    """
    Clase para el procesamiento en hilos de la simulación.
    Esto permite tener el proceso de simulación en paralelo y no interrumpir toda la aplicación.

    Parámetros
    ----------

    - `parent`
        `QObject` que representa el objeto padre donde se instanciará el hilo.

    - `path`
        `str` que representa la ruta del archivo de entrada.

    - `diagnosticos`
        `list` que contiene los nombres de los diagnósticos.

    - `porcientos`
        `list` que contiene los porcientos destinados para cada diagnóstico.

    Responsabilidades
    -----------------

    - `run(self)`: Inicia la simulación. Al llamar a la función `start()`, directamente se llama a esta función.
    - `stop(self)`: Detiene la simulación. Pone fin al hilo de la simulación.
    """

    signal_progBarr = QtCore.pyqtSignal(int)
    signal_terminated = QtCore.pyqtSignal(bool)

    def __init__(self, parent: QObject, path: str, diagnosticos, porcientos) -> None:
        super(Simulation_Thread, self).__init__(parent)
        self.index = 0
        self.is_running = True
        self.env = Environment()
        self.path = path
        self.diagnosticos = diagnosticos
        self.porcientos = porcientos

    # def run(self):
    #     print("Comenzando simulación...")
    #     uci_run = Uci(self.env, self.path, self.diagnosticos, self.porcientos)
    #     self.env.run()
    #     proceso = 0
    #     t_comienzo = time.time()
    #     while True:
    #         proceso += self.env.now
    #         print(proceso)
    #         time.sleep(0.05)
    #         if proceso > 100:
    #             t_final = time.time()
    #             print(f"La simulación terminó a los {(t_final - t_comienzo):.2f} seg.")
    #             self.signal_terminated.emit(True)
    #             break
    #         self.signal_progBarr.emit(proceso / 18864 * 100)
    #     uci_run.exportar_datos()

    def run(self):
        print("Comenzando simulación...")
        proceso = 0
        t_comienzo = time.time()
        while True:
            proceso += 1
            self.signal_progBarr.emit(proceso)
            time.sleep(0.05)
            if proceso > 100:
                t_final = time.time()
                print(f"La simulación terminó a los {(t_final - t_comienzo):.2f} seg.")
                self.signal_terminated.emit(True)
                break
            # self.signal_progBarr.emit(proceso / 18864 * 100)
        # uci_run.exportar_datos()

    def stop(self):
        print("Deteniendo la simulación....")
        self.is_running = False
        self.terminate()
