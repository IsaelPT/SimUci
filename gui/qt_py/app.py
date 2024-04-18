import sys
from PyQt5.QtWidgets import QApplication
from gui.qt_py.main_window import MainWindow


class Application(QApplication):
    """
    Clase Aplicación. Inicia la aplicación.

    Responsabilidades
    -----------------

    - `run()`: Da inicio a la aplicación invocando a la ventana del Menú Principal.
    """

    def __init__(self) -> None:
        super().__init__(sys.argv)
        self.main_win = MainWindow()

    def run(self):
        self.main_win.show()
        sys.exit(self.exec_())


if __name__ == "__main__":
    app = Application()
    app.run()
