from PyQt5.QtWidgets import QMainWindow
from PyQt5.uic import loadUi
from gui.utils.constantes import Rutas
from gui.qt_py.simulation_window import SimulationWindow


class MainWindow(QMainWindow):
    """
    Es la Ventana Menú Principal de la aplicación.

    Responsabilidades
    -----------------

    - `abrir_ventana_simulacion(self)`: Abre una ventana para realizar simulaciones con datos.
    - `cerrar_ventana_simulacion(self)`: Cierra la aplicación.
    """

    def __init__(self) -> None:
        super().__init__()
        loadUi(Rutas.MAINWINDOW_UI, self)  # baseinstance: MainWindow

        # Conexiones de los componentes.
        self.pB_simulacion.clicked.connect(self.abrir_ventana_simulacion)
        self.pB_salir.clicked.connect(self.cerrar_app)

    def abrir_ventana_simulacion(self) -> None:
        """
        Abre la ventana de Simulaciones.
        """

        try:
            self.simulation_win = SimulationWindow(self)
            self.simulation_win.show()
            self.hide()
        except Exception as e:
            print(f"Error al abrir la ventana: {e}")

    def cerrar_ventana_simulacion(self):
        try:
            self.simulation_win.close()
            self.show()
        except Exception as e:
            print(f"Error al cerrar la ventana: {e}")

    def cerrar_app(self):
        try:
            self.close()
        except Exception as e:
            print(f"Error al cerrar la aplicación: {e}")
