from PyQt5.QtWidgets import QMainWindow, QMessageBox
from PyQt5.uic import loadUi
from constantes import Rutas
from simulation_window import SimulationWindow


class MainWindow(QMainWindow):
    """QtWidget que es la Ventana Principal de la aplicación que primeramente sale al iniciar el programa."""

    def __init__(self) -> None:
        super().__init__()
        self.main_win = QMainWindow()
        loadUi(Rutas.ARCHIVO_UI_MAINWINDOW, self.main_win)

        # Conexiones de los componentes.
        self.main_win.pB_simulacion.clicked.connect(self.abrir_ventana_simulacion)
        self.main_win.pB_salir.clicked.connect(self.cerrar_app)

    def abrir_ventana_simulacion(self) -> None:
        """Abre la ventana de Simulaciones."""

        try:
            self.sim_win = SimulationWindow(self)
            loadUi(Rutas.RUTA_ARCHIVO_UI_SIMULATIONWIDGET, self.sim_win)
            self.sim_win.show()
            self.main_win.hide()
        except Exception as e:
            print(f"{e}")
            QMessageBox.warning(self, "Error inesperado", f"Se produjo un error: {e}")

    def cerrar_ventana_simulacion(self):
        self.sim_win.close()

    def cerrar_app(self):
        self.main_win.close()

    def run(self):
        self.main_win.show()
