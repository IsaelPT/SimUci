import sys

from PyQt5.QtWidgets import QApplication

from qt_py.main_window import MainWindow


class Application(QApplication):
    """
    Clase Principal. Le  inicio a la aplicación:
    Cuando se ejecute esta clase, se inicia la aplicación trayendo la ventana del Menú Principal.
    """

    def __init__(self) -> None:
        super().__init__(sys.argv)
        self.main_win = MainWindow()

    def run(self):
        """Muestra la Ventana Principal e inicia la aplicación."""

        self.main_win.show()
        sys.exit(self.exec_())


if __name__ == "__main__":
    app = Application()
    app.run()
