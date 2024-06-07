import sys

from PyQt5.QtWidgets import QApplication

from ui_code.mainmenu_win import MainMenuWindow


class Application(QApplication):
    def __init__(self) -> None:
        super().__init__(sys.argv)
        self.main_win = MainMenuWindow()

    def run(self):
        self.main_win.show()
        sys.exit(self.exec_())


if __name__ == "__main__":
    app = Application()
    app.run()
