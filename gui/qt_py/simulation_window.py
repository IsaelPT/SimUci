import time
import pandas as pd
from PyQt5.QtCore import QThread, pyqtSignal
from PyQt5.QtWidgets import QWidget, QFileDialog
from PyQt5.QtGui import QStandardItem, QStandardItemModel
from PyQt5.uic import loadUi
from constantes import Rutas, Mensajes


class SimulationWindow(QWidget):

    def __init__(self, main_win) -> None:
        super().__init__()
        self.main_win = main_win
        self.sim_win = QWidget()
        loadUi(Rutas.RUTA_ARCHIVO_UI_SIMULATIONWIDGET, self.sim_win)

        # QListView para visualizar los datos.
        self.sim_win.listView_model = QStandardItemModel()
        self.sim_win.listView_simulacion.setModel(self.listView_model)  # QListView()

        # QLineEdit para visualizar la dirección donde se localiza el archivo.
        self.sim_win.lineEdit_ruta_datos.setText(Mensajes.MENSAJE_LINE_EDIT)

        # Conexiones de los componentes
        self.sim_win.pB_cargar.clicked.connect(self.cargar_csv)
        self.sim_win.pB_comenzar.clicked.connect(self.comenzar_simulacion)
        self.sim_win.pB_detener.clicked.connect(self.detener_simulacion)
        self.sim_win.pB_salir.clicked.connect(self.cerrar_ventana)

        # self.setup_tabla_diagnosticos()

    def cargar_csv(self) -> None:
        """Carga el archivo .csv a través de un QFileDialog."""

        nombre_archivo, _ = QFileDialog.getOpenFileName(
            self, "Abrir CSV", "", "Archivos CSV (*.csv)"
        )

        if nombre_archivo:
            try:
                self.datos = pd.read_csv(nombre_archivo)
                print(f"Archivo CSV '{nombre_archivo}' cargado correctamente.")
                self.lineEdit_ruta_datos.setText(nombre_archivo)
            except Exception as e:
                print(f"{e}")

    def comenzar_simulacion(self) -> None:
        """Da inicio a la simulación al ser pulsado el botón 'Comenzar Simulación'."""

        try:
            self.pB_comenzar.setEnabled(False)
            # TODO: Ver exactamente aquí la situación con los hilos, algo está mal.
            self.hilo_simulacion = HiloSimulacion(self)
            self.hilo_simulacion.start()
            # self.hilo_simulacion.wait()
            self.hilo_simulacion.quit()
            print("Finalizó la simulación.")
            self.pB_comenzar.setEnabled(True)
        except Exception as e:
            print(f"La simulación terminó con el siguiente error: {e}")

    def detener_simulacion(self) -> None:
        """Detiene la simulación a medio proceso."""

        if self.hilo_simulacion.isRunning():
            self.hilo_simulacion.wait()

    def cerrar_ventana(self):
        """Cierra esta ventana de Simulación."""

        try:
            self.main_win.show()
        except Exception as e:
            print(f"Ocurrió un error ({e}) al tratar de cerrar la ventana.")

    def actualizar_proceso(self, valor) -> None:
        """Modifica el valor de la Barra de Progreso de Simulación y el de la tabla de simulación a través de informacións."""

        self.progressBar.setValue(valor)
        self.agregar_elemento_listView(f"Proceso: {valor}")

    def agregar_elemento_listView(self, texto):
        """Agrega elementos (texto) a la ListView en la ventana de simulación para tener información sobre el proceso de simulación."""

        item = QStandardItem(texto)
        self.listView_model.appendRow(item)

    def setup_tabla_diagnosticos(self):
        """Define el modelo del QColumnView para visualizar los diagnosticos y % que se obtienen de los pacientes."""

        # TODO: Datos de ejemplo
        FILAS = 10
        COLUMNAS = 2

        modelo_tabla = QStandardItemModel(FILAS, COLUMNAS)

        modelo_tabla.setHorizontalHeaderItem(0, QStandardItem("Diagnosticos"))
        modelo_tabla.setHorizontalHeaderItem(1, QStandardItem("Porcientos"))

        for fila in range(FILAS):
            diagnostico = QStandardItem(f"Diagnostico {fila + 1}")
            porciento = QStandardItem(f"Porciento {fila + 1}")
            modelo_tabla.setItem(fila, 0, diagnostico)
            modelo_tabla.setItem(fila, 1, porciento)

        # Definir el model del QTableView para visualizar los diagnosticos en 'simulation_window'.
        self.table_view = self.tableView_diagnosticos
        self.table_view.setModel(modelo_tabla)
        self.table_view.resizeColumnsToContents()

    def limpiar_datos_simulationWindow(self):
        """Limpia los datos de los campos presentes en el QWidget de 'simulation_window'"""

        # Limpia los datos del modelo de la QListView que muestra datos de simulación.
        if self.listView_model is not None:
            self.listView_model.clear()

        # Vuelve a 0 el progreso de la barra de progreso de simulación.
        self.progressBar.setValue(0)


# TODO: WIP.
class HiloSimulacion(QThread):
    """Clase hilo para procesar la simulación en paralelo."""

    progreso_barra = pyqtSignal(int)

    def __init__(self, application):
        super().__init__()
        self.application = application

    def simular(self):
        print("Comenzó la simulación.")

        for valor in range(0, 101, 10):
            self.progreso_barra.emit(valor)
            self.application.actualizar_proceso(valor)
            time.sleep(0.2)

    def run(self):
        self.simular()
