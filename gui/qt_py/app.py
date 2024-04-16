import sys
from PyQt5.QtWidgets import QApplication
from main_window import MainWindow


class Application(QApplication):
    """Clase que inicia la aplicación. Invoca al menú principal (MainMenu) y sus componentes."""

    def __init__(self) -> None:
        self.app = QApplication(sys.argv)

    def run(self):
        self.main_win = MainWindow()
        self.main_win.run()
        sys.exit(self.app.exec_())


if __name__ == "__main__":
    app = Application()
    app.run()
