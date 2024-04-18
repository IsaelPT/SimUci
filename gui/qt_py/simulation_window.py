import pandas as pd
from PyQt5.QtWidgets import QWidget, QFileDialog
from PyQt5.QtGui import QStandardItem, QStandardItemModel
from PyQt5.uic import loadUi
from utils.constantes import Rutas


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
    """

    def __init__(self, main_win) -> None:
        super().__init__()
        self.main_win = main_win
        loadUi(Rutas.SIMULATIONWIDGET_UI, self)  # baseinstance: SimulationWindow

        # QLineEdit para visualizar la dirección donde se localiza el archivo.
        self.lineEdit_ruta_datos.setText("Ruta de archivo...")

        # QColumnView para visualizar los diagnosticos / porcientos.
        modelo_tabla = QStandardItemModel(0, 2)
        modelo_tabla.setHorizontalHeaderItem(0, QStandardItem("Diagnosticos"))
        modelo_tabla.setHorizontalHeaderItem(1, QStandardItem("Porcientos"))
        self.table_view = self.tableView_diagnosticos
        self.table_view.setModel(modelo_tabla)
        self.table_view.resizeColumnsToContents()

        # self.init_tabla_diagnosticos()

        # Conexiones de los componentes.
        self.pB_cargar.clicked.connect(self.cargar_csv)
        self.pB_comenzar.clicked.connect(self.comenzar_simulacion)
        self.pB_detener.clicked.connect(self.detener_simulacion)
        self.pB_salir.clicked.connect(self.cerrar_ventana)

    def cargar_csv(self) -> None:
        """
        Carga el archivo .csv a través de un `QFileDialog`.
        """

        ruta_archivo, _ = QFileDialog.getOpenFileName(
            self, "Abrir CSV", "", "Archivos CSV (*.csv)"
        )

        if ruta_archivo:
            try:
                self.datos = pd.read_csv(ruta_archivo)
                self.lineEdit_ruta_datos.setText(ruta_archivo)
                print(f"Archivo CSV '{ruta_archivo}' cargado correctamente!")
            except Exception as e:
                print(f"{e}")

    def comenzar_simulacion(self) -> None:
        """
        Da inicio a la simulación al ser pulsado el botón "Comenzar Simulación".
        """

        try:
            print("Se presionó el botón de 'Comenzar Simulacion'.")
        except Exception as e:
            print(f"Ocurrió un error inesperado: {e}")
        # try:
        #     self.pB_comenzar.setEnabled(False)
        #     # TODO: Ver exactamente aquí la situación con los hilos, algo está mal.
        #     self.hilo_simulacion = HiloSimulacion(self)
        #     self.hilo_simulacion.start()
        #     # self.hilo_simulacion.wait()
        #     self.hilo_simulacion.quit()
        #     print("Finalizó la simulación.")
        #     self.pB_comenzar.setEnabled(True)
        # except Exception as e:
        #     print(f"La simulación presentó siguiente error: {e}")

    def detener_simulacion(self) -> None:
        """
        Detiene la simulación a medio proceso.
        """

        try:
            print("Se presionó el botón de 'Detener Simulación'.")
        except Exception as e:
            print(f"Ocurrió un error inesperado: {e}")
        # if self.hilo_simulacion.isRunning():
        #     self.hilo_simulacion.wait()

    def cerrar_ventana(self):
        """
        Cierra esta ventana de Simulación.
        """

        try:
            self.close()
            self.main_win.show()
        except Exception as e:
            print(f"Ocurrió un error al cerrar la ventana: {e}")

    def init_tabla_diagnosticos(self):
        """
        Define el modelo del `QColumnView` para visualizar los diagnosticos y % que se obtienen de los pacientes.
        """

        FILAS = 10
        COLUMNAS = 2

        modelo_tabla = QStandardItemModel(FILAS, COLUMNAS)
        modelo_tabla.setHorizontalHeaderItem(0, QStandardItem("Diagnosticos"))
        modelo_tabla.setHorizontalHeaderItem(1, QStandardItem("Porcientos"))

        def insertar_lista(diagnosticos: list, porcientos: list) -> None:
            pass

        for fila in range(FILAS):
            diagnostico = QStandardItem(f"Diagnostico {fila + 1}")
            porciento = QStandardItem(f"Porciento {fila + 1}")
            modelo_tabla.setItem(fila, 0, diagnostico)
            modelo_tabla.setItem(fila, 1, porciento)

        # Definir el model del QTableView para visualizar los diagnosticos en 'simulation_window'.
        self.table_view = self.tableView_diagnosticos
        self.table_view.setModel(modelo_tabla)
        self.table_view.resizeColumnsToContents()


# # TODO: WIP.
# class HiloSimulacion(QThread):
#     """Clase hilo para procesar la simulación en paralelo."""

#     progreso_barra = pyqtSignal(int)

#     def __init__(self, application):
#         super().__init__()
#         self.application = application

#     def simular(self):
#         print("Comenzó la simulación.")

#         for valor in range(0, 101, 10):
#             self.progreso_barra.emit(valor)
#             self.application.actualizar_proceso(valor)
#             time.sleep(0.2)

#     def run(self):
#         self.simular()
