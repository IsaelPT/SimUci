import time
import pandas as pd

from PyQt5 import QtCore
from PyQt5.QtCore import QObject, QThread
from PyQt5.QtWidgets import QWidget, QFileDialog, QMessageBox
from PyQt5.QtGui import QStandardItem, QStandardItemModel
from PyQt5.uic import loadUi
from simpy import Environment

from qt_py.constantes import Rutas
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
    - `init_tabla_diagnosticos():`
    """

    ruta_archivo_csv: str = None

    def __init__(self, main_win) -> None:

        super().__init__()
        self.main_win = main_win
        loadUi(Rutas.SIMULATIONWIDGET_UI, self)  # baseinstance: SimulationWindow

        # QLineEdit para visualizar la dirección donde se localiza el archivo.
        self.lineEdit_ruta_datos.setText("Ruta de archivo...")

        # QColumnView para visualizar los diagnosticos / porcientos.
        self.modelo_tabla = QStandardItemModel(0, 2)
        self.modelo_tabla.setHorizontalHeaderItem(0, QStandardItem("Diagnosticos"))
        self.modelo_tabla.setHorizontalHeaderItem(1, QStandardItem("Porcientos"))
        # self.tableView_diagnosticos = self.tableView_diagnosticos
        self.tableView_diagnosticos.setModel(self.modelo_tabla)
        self.tableView_diagnosticos.resizeColumnsToContents()

        self.thread = {}  # Hilos de esta ventana.

        # Conexiones de los componentes.
        self.pB_cargar.clicked.connect(self.cargar_csv)
        self.pB_comenzar.clicked.connect(self.comenzar_simulacion)
        self.pB_detener.clicked.connect(self.detener_simulacion)
        self.pB_salir.clicked.connect(self.cerrar_ventana)

    def cargar_csv(self) -> None:
        """
        Carga el archivo .csv a través de un `QFileDialog`.
        """

        self.ruta_archivo_csv, _ = QFileDialog.getOpenFileName(
            self, "Abrir CSV", "", "Archivos CSV (*.csv)"
        )

        if self.ruta_archivo_csv:
            try:
                self.datos_csv = pd.read_csv(self.ruta_archivo_csv)
                diagnosticos = proc_d.get_diagnostico_list(self.ruta_archivo_csv)
                self._init_tabla_diagnosticos(diagnosticos)  # Ingresar datos en Tabla.
                self.lineEdit_ruta_datos.setText(self.ruta_archivo_csv)
                print(f"Archivo CSV '{self.ruta_archivo_csv}' cargado correctamente!")
            except Exception as e:
                print(f"Ocurrió un error inesperado: {e}")
        else:
            raise Exception("La ruta de archivos no existe u ocurrió un error.")

    def comenzar_simulacion(self) -> None:
        """
        Da inicio a la simulación al ser pulsado el botón "Comenzar Simulación".
        """

        try:
            print("-- Se presionó el botón de 'Comenzar Simulacion' --")
            self.thread[1] = Simulation_Thread(self)
            self.thread[1].run(
                self.ruta_archivo_csv,
                proc_d.get_diagnostico_list,
                self.modelo_tabla.takeColumn(1),
            )
            self.pB_cargar_csv.setEnnabled(False)
            self.pB_comenzar.setEnabled(False)
            self.thread[1].signal_progBarr.connect(self._update_progress_bar)
            self.thread[1].signal_terminated.connect(self.pB_comenzar.setEnabled)
        except Exception as e:
            print(f"Ocurrió un error inesperado a la hora de correr la simulación: {e}")

    def detener_simulacion(self) -> None:
        """
        Detiene la simulación a medio proceso.
        """

        try:
            print("Se presionó el botón de 'Detener Simulación'.")
            self.thread[1].stop()
            self.progressBar.setValue(0)
            self.pB_cargar_csv.setEnnabled(True)
            self.pB_comenzar.setEnabled(True)
            QMessageBox().warning(
                self, "Detención de simulación", "Se ha detenido la simulación."
            )
        except Exception as e:
            print(f"Ocurrió un error inesperado: {e}")

    def cerrar_ventana(self) -> None:
        """
        Cierra esta ventana de Simulación.
        """

        try:
            self.close()
            self.main_win.show()
        except Exception as e:
            print(f"Ocurrió un error al cerrar la ventana: {e}")

    def _init_tabla_diagnosticos(self, diagnosticos) -> None:
        """
        Define el modelo del `QColumnView` para visualizar los diagnosticos y porcientos
        que se obtienen de los pacientes.
        """

        FILAS = len(diagnosticos)
        print(f"Cantidad de diagnosticos: {FILAS}")
        print(f"Lista de diagnosticos importada:\n{diagnosticos}")

        for d in diagnosticos:
            item = QStandardItem(d)
            self.modelo_tabla.appendRow(item)

        self.tableView_diagnosticos.resizeColumnsToContents()

    def _update_progress_bar(self, contador):
        self.progressBar.setValue(contador)


class Simulation_Thread(QThread):
    """
    Clase para el procesamiento en hilos de la simulación.
    Esto permite tener el proceso de simulación en paralelo y no interrumpir toda la aplicación.

    Responsabilidades
    -----------------

    - `start()`: Inicia la simulación. Al llamar a la función `start()`, directamente se llama a esta función.
    - `run()`: Detiene la simulación. Pone fin al hilo de la simulación.
    """

    signal_progBarr = QtCore.pyqtSignal(int)
    signal_terminated = QtCore.pyqtSignal(bool)

    def __init__(self, parent: QObject | None = ...) -> None:
        super(Simulation_Thread, self).__init__(parent)
        self.index = 0
        self.is_running = True
        self.env = Environment()

    # Nota: Cuando se llama a `start()` se llama directamente a esta función: `run()`
    def run(self, path, diagnosticos, porcientos):
        print("Comenzando simulación...")
        uci_run = Uci(self.env, path, diagnosticos, porcientos)
        self.env.run()
        proceso = 0
        t_comienzo = time.time()
        while True:
            proceso += self.env.now
            time.sleep(0.05)
            if proceso == 101:
                t_final = time.time()
                print(f"La simulación terminó a los {(t_final - t_comienzo):.2f} seg.")
                self.signal_terminated.emit(True)
                break
            self.signal_progBarr.emit(proceso / 18864 * 100)
        uci_run.exportar_datos()

    def stop(self):
        print("Deteniendo la simulación....")
        self.is_running = False
        self.terminate()
