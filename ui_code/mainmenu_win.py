import traceback

from PyQt5.QtGui import QIcon
from PyQt5.QtWidgets import QMainWindow
from PyQt5.uic import loadUi

from ui_code.simulation_win import SimulationWindow
from tools.excepciones import ExceptionSaver
from tools.utils import Rutas


class MainMenuWindow(QMainWindow):
    simulation_win: 'SimulationWindow' = None

    def __init__(self) -> None:
        super().__init__()
        loadUi(Rutas.UiFiles.MAINWINDOW_UI, self)  # baseinstance: MainWindow
        self._init_fields()
        self._init_components()
        self._connect_signals()

    def _init_fields(self):
        pass

    def _init_components(self) -> None:
        self.setWindowIcon(QIcon(Rutas.Iconos.WINDOWICON_HEALTH))

    def _connect_signals(self) -> None:
        self.actionVentana_simulacion.triggered.connect(self.abrir_ventana_simulacion)
        self.pb_simulacion.clicked.connect(self.abrir_ventana_simulacion)
        self.pb_salir.clicked.connect(self.cerrar_app)

    def abrir_ventana_simulacion(self) -> None:
        try:
            self.simulation_win = SimulationWindow(self)
            self.simulation_win.show()
        except Exception as e:
            print(f"Error al abrir la ventana de simulación: {e}\n{traceback.format_exc()}")
            ExceptionSaver().save(e)

    def cerrar_ventana_simulacion(self) -> None:
        try:
            self.simulation_win.close()
            self.show()
        except Exception as e:
            print(f"Error al cerrar la ventana de simulación: {e}\n{traceback.format_exc}")
            ExceptionSaver().save(e)

    def cerrar_app(self) -> None:
        try:
            self.close()  # Cerrar Main Menu Window.
        except Exception as e:
            print(f"Error al cerrar la aplicación: {e}\n{traceback.format_exc()}")
            ExceptionSaver().save(e)
