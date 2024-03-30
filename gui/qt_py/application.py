import time
import sys
import pandas as pd
from PyQt5.QtCore import QThread, pyqtSignal
from PyQt5.QtWidgets import (
    QMainWindow,
    QApplication,
    QWidget,
    QMessageBox,
    QFileDialog,
    QLineEdit,
)
from PyQt5.QtGui import QStandardItemModel, QStandardItem
from PyQt5.uic import loadUi


class MainApplication:
    """Clase que inicializa la aplicación trayendo la ventana principal y sus componentes."""

    ARCHIVO_UI_MAINWINDOW = "gui/qt_ui/main_window.ui"
    ARCHIVO_UI_SIMULATIONWIDGET = "gui/qt_ui/simulation_widget.ui"

    MENSAJE_LINE_EDIT = "Ruta del archivo..."

    def __init__(self) -> None:

        # Definir la Aplicación Principal.
        self.app = QApplication(sys.argv)

        # Definir la Ventana Principal.
        self.main_window = QMainWindow()
        loadUi(self.ARCHIVO_UI_MAINWINDOW, self.main_window)

        # Definir el QWidget para Simulación.
        self.simulation_window = QWidget()
        loadUi(self.ARCHIVO_UI_SIMULATIONWIDGET, self.simulation_window)
        self.simulation_window.lineEdit_ruta_datos.setText(self.MENSAJE_LINE_EDIT)

        # TODO: Esta inicialización del modelo quizás trasladarla luego a otra parte del código.
        # Definir el modelo del QListView para visualizar los datos en 'simulation_window'.
        self.modelo_listView = QStandardItemModel()
        self.simulation_window.listView_simulacion.setModel(self.modelo_listView)

        # Definir el lineEdit donde se visualiza la dirección donde se lcoaliza el archivo.
        self.linea_de_ruta = QLineEdit()
        self.linea_de_ruta = self.simulation_window.lineEdit_ruta_datos

        # Conexiones de los componentes de 'main_window'
        self.main_window.pushButton_simulacion.clicked.connect(
            self.abrir_ventana_simulacion
        )
        self.main_window.pushButton_salir.clicked.connect(self.cerrar_aplicacion)

        # Conexiones de los componentes de 'simulation_window'
        self.simulation_window.pushButton_salir.clicked.connect(
            self.cerrar_ventana_simulacion
        )
        self.simulation_window.pushButton_comenzar.clicked.connect(
            self.comenzar_simulacion
        )
        self.simulation_window.pushButton_detener.clicked.connect(
            self.detener_simulacion
        )
        self.simulation_window.pushButton_cargar.clicked.connect(self.cargar_csv)

    def cerrar_aplicacion(self):
        """
        - main_window

        Cierra la aplicacióm.
        """
        self.app.quit()

    def abrir_ventana_simulacion(self) -> None:
        """
        Abre la ventana de simulación.

        Los componentes de dicha ventana se cargan de un archivo .ui a través de una ruta.
        """

        try:
            if self.simulation_window is not None:
                self.simulation_window.show()
                self.setup_tabla_diagnosticos()
                self.main_window.hide()
                print("Se abrió la ventana de simulación.")
            else:
                print("No se pudo abrir la ventana de simulación.")

        except Exception as e:
            print(f"{e}")
            QMessageBox.warning(
                self.main_window, "Error inesperado", f"Se produjo un error: {e}"
            )

    def cerrar_ventana_simulacion(self) -> None:
        """
        - simulation_window

        Detiene la simulación.
        """
        try:
            self.simulation_window.close()
            print("Se cerró la ventana de simulación.")
            if self.linea_de_ruta is not None:
                self.simulation_window.lineEdit_ruta_datos.setText(
                    self.MENSAJE_LINE_EDIT
                )
            self.main_window.show()
        except Exception as e:
            print(f"{e}")
            QMessageBox.warning(
                self.simulation_window, "Cerrar simulación", f"Se produjo un error: {e}"
            )

    def comenzar_simulacion(self) -> None:
        """
        - simulation_window

        Da inicio a la simulación.

        Esta simulación comienza al ser pulsado el botón 'Comenzar Simulación'
        """

        try:
            self.simulation_window.pushButton_comenzar.setEnabled(False)

            self.hilo_simulacion = HiloSimulacion(self)

            # TODO: Ver exactamente aquí la situación con los hilos, algo está mal.
            self.hilo_simulacion.start()
            # self.hilo_simulacion.wait()
            self.hilo_simulacion.quit()

            print("Finalizó la simulación.")
            self.simulation_window.pushButton_comenzar.setEnabled(True)
        except Exception as e:
            print(f"La simulación terminó con el siguiente error: {e}")

    # TODO!: Detener la simulación correctamente y trabajar el asunto de los hilos.
    def detener_simulacion(self) -> None:
        """
        - simulation_window

        Detiene la simulación a medio proceso.
        """
        if self.hilo_simulacion.isRunning():
            self.hilo_simulacion.wait()

    def cargar_csv(self) -> None:
        """
        - simulation window

        Carga el archivo .csv a través de un QFileDialog.
        """

        nombre_archivo, _ = QFileDialog.getOpenFileName(
            self.main_window, "Abrir CSV", "", "Archivos CSV (*.csv)"
        )

        if nombre_archivo:
            try:
                self.datos = pd.read_csv(nombre_archivo)
                print(
                    f"Archivo CSV '{nombre_archivo}' cargado exitosamente. Filas totales: {len(self.datos)}"
                )
                self.simulation_window.lineEdit_ruta_datos.setText(nombre_archivo)
            except Exception as e:
                QMessageBox.warning(
                    self.main_window,
                    "Error al cargar el CSV",
                    f"Se produjo un error al cargar el archivo CSV: {e}",
                )
            print("Se cargó el Archivo correctamente.")

    def actualizar_proceso(self, valor) -> None:
        """
        - simulation_window

        Modifica el valor de la Barra de Progreso de Simulación y el de la tabla de simulación a través de informacións.
        """

        self.simulation_window.progressBar.setValue(valor)
        self.agregar_elemento_listView(f"Proceso: {valor}")

    def agregar_elemento_listView(self, texto):
        """
        - simulation_window

        Agrega elementos (texto) a la ListView en la ventana de simulación para tener información sobre el proceso de simulación.
        """

        item = QStandardItem(texto)
        self.modelo_listView.appendRow(item)

    def setup_tabla_diagnosticos(self):
        """Definir el modelo del QColumnView para visualizar los diagnosticos y % que se obtienen de los pacientes."""

        FILAS = 10
        COLUMNAS = 2

        modelo_tabla = QStandardItemModel(FILAS, COLUMNAS)

        modelo_tabla.setHorizontalHeaderItem(0, QStandardItem("Diagnosticos"))
        modelo_tabla.setHorizontalHeaderItem(1, QStandardItem("Porcientos"))

        for fila in range(FILAS):
            for columna in range(COLUMNAS):
                item = QStandardItem(f"Diagnóstico {fila}, Porciento {columna + 1}")
                modelo_tabla.setItem(fila, columna, item)

        # Definir el model del QTableView para visualizar los diagnosticos en 'simulation_window'.
        self.table_view = self.simulation_window.tableView_diagnosticos
        self.table_view.setModel(modelo_tabla)
        self.table_view.resizeColumnsToContents()

    def limpiar_datos_simulationWindow(self):
        """
        - simulation_window

        Limpia los datos de los campos presentes en el QWidget de 'simulation_window'
        """

        # Limpia los datos del modelo de la QListView que muestra datos de simulación.
        if self.modelo_listView is not None:
            self.modelo_listView.clear()

        # Vuelve a 0 el progreso de la barra de progreso de simulación.
        self.simulation_window.progressBar.setValue(0)

    def run(self):
        self.main_window.show()
        sys.exit(self.app.exec_())


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


if __name__ == "__main__":
    app = MainApplication()
    app.run()
