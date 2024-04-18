from PyQt5.QtWidgets import QMainWindow, QMessageBox
from PyQt5.uic import loadUi
from constantes import Rutas
from simulation_window import SimulationWindow


class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        loadUi(Rutas.ARCHIVO_UI_MAINWINDOW, self)  # Cargar la UI en la ventana principal 'self'

        # Conexiones de los componentes
        self.pB_simulacion.clicked.connect(self.abrir_ventana_simulacion)
        self.pB_salir.clicked.connect(self.cerrar_app)

    def abrir_ventana_simulacion(self):
        try:
            self.sim_win = SimulationWindow(self)
            self.sim_win.show()
            self.hide()  # Ocultar la ventana principal
        except Exception as e:
            print(f"{e}")
            QMessageBox.warning(self, "Error inesperado", f"Se produjo un error: {e}")

    def cerrar_ventana_simulacion(self):
        self.sim_win.close()

    def cerrar_app(self):
        self.close()

    def run(self):
        self.show()
