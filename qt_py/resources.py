import typing

from PyQt5.QtWidgets import QMessageBox
from PyQt5.QtGui import QIcon, QPixmap


class Rutas:
    """Para variables con rutas estáticas para cargar componentes necesarios."""

    class Ui_Files:
        """Donde se localizan los archivos `.ui` que sirven para construir las GUI."""

        MAINWINDOW_UI = "qt_ui/main_window_v2.ui"
        SIMULATIONWIDGET_UI = "qt_ui/simulation_widget.ui"

    class Iconos:
        """Donde se localizan los archivos de iconos."""

        # fmt: off
        WINDOWICON_HEALTH = "ui_utils/icons/MageHealthSquareFill.png"
        WINDOWICON_SIMULATION = "ui_utils/icons/FluentMdl2TestPlan.png"
        ICON_QMESSAGEBOX_OK = "ui_utils/icons/FlatColorIconsOk.png"
        ICON_QMESSAGEBOX_STOP = "ui_utils/icons/MaterialSymbolsStopCircleOutlineRounded.png"
        ICON_QMESSAGEBOX_SIMULATIONBREAK = "ui_utils/icons/CodiconBeakerStop.png"
        ICON_QMESSAGEBOX_TIME = "ui_utils/icons/PhTimerDuotone.png"
        # fmt: on


class Estilos:
    """
    Contiene estilos personalizados que se utilizan en botones y otros componentes
    en las diversas interfaces gráficas.
    """

    botones = {
        "botones_acciones_verdes": """
            QPushButton {
                background-color: #b2f2bb;
                border: 1px solid #8f8f91;
                border-radius: 5px;
            }
            QPushButton:hover {
                background-color: #8f8f91;
            }
        """,
    }
    """Para botones `Comenzar Simulacion`, `Detener Simulacion`, `Salir` y otros."""


class CustomQMessageBox:
    """
    Clase que permite crear mensajes de error personalizados.
    """

    QWidget = typing.TypeVar("QWidget")

    def finalizado(
        parent: typing.Optional[QWidget] = None,
        titulo: typing.Optional[str] = "Finalizado",
        mensaje: typing.Optional[str] = "Proceso finalizado.",
    ) -> None:
        qmsgb = QMessageBox(parent)
        qmsgb.setWindowTitle(titulo)
        qmsgb.setText(mensaje)
        qmsgb.setWindowIcon(QIcon(Rutas.Iconos.ICON_QMESSAGEBOX_OK))
        qmsgb.setIconPixmap(QPixmap(Rutas.Iconos.ICON_QMESSAGEBOX_TIME))
        qmsgb.setStandardButtons(QMessageBox.Ok)
        qmsgb.setDefaultButton(QMessageBox.Ok)
        qmsgb.exec_()

    def interrupcion(
        parent: typing.Optional[QWidget] = None,
        titulo: typing.Optional[str] = "Interrupción",
        mensaje: typing.Optional[str] = "Proceso interrumpido.",
    ) -> None:
        qmsgb = QMessageBox(parent)
        qmsgb.setWindowTitle(titulo)
        qmsgb.setText(mensaje)
        qmsgb.setWindowIcon(QIcon(Rutas.Iconos.ICON_QMESSAGEBOX_STOP))
        qmsgb.setIconPixmap(QPixmap(Rutas.Iconos.ICON_QMESSAGEBOX_SIMULATIONBREAK))
        qmsgb.setStandardButtons(QMessageBox.Ok)
        qmsgb.setDefaultButton(QMessageBox.Ok)
        qmsgb.exec_()
