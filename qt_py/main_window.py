import traceback

from PyQt5.QtGui import QCloseEvent
from PyQt5.QtWidgets import QMainWindow
from PyQt5.QtGui import QIcon
from PyQt5.uic import loadUi

from qt_py.simulation_window import SimulationWindow
from qt_py.resources import Rutas, Estilos


class MainWindow(QMainWindow):
    """
    Es la Ventana del Menú Principal de la aplicación.
    Desde esta ventana se acceden a las diferentes opciones que dispone la aplicación.
    """

    simulation_win: SimulationWindow = None

    def __init__(self) -> None:
        super().__init__()
        loadUi(Rutas.Ui_Files.MAINWINDOW_UI, self)  # baseinstance: MainWindow

        # Conexiones de los componentes.
        self.actionVentana_simulacion.triggered.connect(self.abrir_ventana_simulacion)
        self.pB_simulacion.clicked.connect(self.abrir_ventana_simulacion)
        self.pB_salir.clicked.connect(self.cerrar_app)

        # Estilos personalizados a los componentes.
        self.pB_simulacion.setStyleSheet(Estilos.botones["botones_acciones_verdes"])
        self.pB_ajustes.setStyleSheet(Estilos.botones["botones_acciones_verdes"])
        self.pB_salir.setStyleSheet(Estilos.botones["botones_acciones_verdes"])

        # Ajustes iniciales a componentes.
        self.setWindowIcon(QIcon(Rutas.Iconos.WINDOWICON_HEALTH))

    def abrir_ventana_simulacion(self) -> None:
        try:
            self.simulation_win = SimulationWindow(self)
            self.simulation_win.show()
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
            self.close()  # Cerrar Main Window.
        except:
            print(f"Error al cerrar la aplicación:\n{traceback.format_exc()}")

    def closeEvent(self, event: QCloseEvent | None) -> None:
        """Método `closeEvent` sobrescrito donde su función es cerrar las ventanas que estén posiblemente
        abiertas en el momento de cerrar esta ventana en Main Window.


        Args:
            `event (QCloseEvent | None)`: Su propósito por el momento es para servir de firma de sobrescritura del método original.
        """

        if self.simulation_win:
            self.simulation_win.cerrar_ventana()
