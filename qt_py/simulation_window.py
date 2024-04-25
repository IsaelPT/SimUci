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
import threading

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

        self.threads = []  # Hilos de esta ventana.

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

        if self.ruta_archivo_csv is not None:
            try:
                #self.datos_csv = pd.read_csv(self.ruta_archivo_csv)
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
            if self.ruta_archivo_csv is not None:
                runner = Uci(self.ruta_archivo_csv,
                             proc_d.get_diagnostico_list(self.ruta_archivo_csv),
                             self._get_porcientos_de_tabla())
                runner.signal.signal_progBarr.connect(self._update_progressBarr)
                runner.signal.signal_terminated.connect(self.pB_comenzar.setEnabled)
                self.pB_comenzar.setEnabled(False)
                self.pB_cargar.setEnabled(False)
                runner.start()
                self.threads.append(runner)
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
            for runner in self.threads:
                runner.stop()
            self.progressBar.setValue(0)
            self.pB_cargar.setEnabled(True)
            self.pB_comenzar.setEnabled(True)
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
            #self.main_win.show()
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

        print(f"Cantidad de diagnosticos: {len(diagnosticos)}")
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

        self.progressBar.setValue(int(contador / 17880 * 100))

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
        for i in range(self.modelo_tabla.rowCount()):
            item_porcentaje = self.modelo_tabla.item(i, 1)
            porcentajes.append(int(item_porcentaje.text()))
        print(f"Lista de porcentajes:\n{porcentajes}")
        return porcentajes