import traceback

from PyQt5.QtWidgets import QMainWindow
from PyQt5.uic import loadUi

from qt_py.simulation_window import SimulationWindow
from qt_py.constantes import Rutas, Estilos


class MainWindow(QMainWindow):
    """
    Es la Ventana Menú Principal de la aplicación.

    Responsabilidades
    -----------------

    - `abrir_ventana_simulacion(self)`: Abre una ventana para realizar simulaciones con datos.
    - `cerrar_ventana_simulacion(self)`: Cierra la ventana de simulación.
    - `cerrar_app(self)`: Cierra la aplicación.
    """

    def __init__(self) -> None:
        super().__init__()
        loadUi(Rutas.MAINWINDOW_UI, self)  # baseinstance: MainWindow

        # Conexiones de los componentes.
        self.actionVentana_simulacion.triggered.connect(self.abrir_ventana_simulacion)
        self.pB_simulacion.clicked.connect(self.abrir_ventana_simulacion)
        self.pB_salir.clicked.connect(self.cerrar_app)

        # Estilos personalizados a los componentes.
        self.pB_simulacion.setStyleSheet(Estilos.botones["botones_acciones_verdes"])
        self.pB_salir.setStyleSheet(Estilos.botones["botones_acciones_verdes"])

    def abrir_ventana_simulacion(self) -> None:
        try:
            self.simulation_win = SimulationWindow(self)
            self.simulation_win.show()
            self.hide()
        except:
            print(f"Error al abrir la ventana de simulación:\n{traceback.format_exc()}")

    def cerrar_ventana_simulacion(self) -> None:
        try:
            self.simulation_win.close()
            self.show()
        except:
            print(f"Error al cerrar la ventana de simulación:\n{traceback.format_exc}")

    def cerrar_app(self) -> None:
        try:
            self.close()
        except:
            print(f"Error al cerrar la aplicación:\n{traceback.format_exc}")
