# -*- coding: utf-8 -*-

################################################################################
## Form generated from reading UI file 'simulation_widget.ui'
##
## Created by: Qt User Interface Compiler version 6.6.2
##
## WARNING! All changes made in this file will be lost when recompiling UI file!
################################################################################

from PySide6.QtCore import (QCoreApplication, QDate, QDateTime, QLocale,
    QMetaObject, QObject, QPoint, QRect,
    QSize, QTime, QUrl, Qt)
from PySide6.QtGui import (QBrush, QColor, QConicalGradient, QCursor,
    QFont, QFontDatabase, QGradient, QIcon,
    QImage, QKeySequence, QLinearGradient, QPainter,
    QPalette, QPixmap, QRadialGradient, QTransform)
from PySide6.QtWidgets import (QApplication, QColumnView, QFrame, QLineEdit,
    QListView, QProgressBar, QPushButton, QSizePolicy,
    QVBoxLayout, QWidget)

class Ui_SimulationWidget(object):
    def setupUi(self, SimulationWidget):
        if not SimulationWidget.objectName():
            SimulationWidget.setObjectName(u"SimulationWidget")
        SimulationWidget.resize(600, 732)
        palette = QPalette()
        brush = QBrush(QColor(0, 0, 0, 255))
        brush.setStyle(Qt.SolidPattern)
        palette.setBrush(QPalette.Active, QPalette.WindowText, brush)
        brush1 = QBrush(QColor(208, 240, 236, 255))
        brush1.setStyle(Qt.SolidPattern)
        palette.setBrush(QPalette.Active, QPalette.Button, brush1)
        brush2 = QBrush(QColor(255, 255, 255, 255))
        brush2.setStyle(Qt.SolidPattern)
        palette.setBrush(QPalette.Active, QPalette.Light, brush2)
        brush3 = QBrush(QColor(231, 247, 245, 255))
        brush3.setStyle(Qt.SolidPattern)
        palette.setBrush(QPalette.Active, QPalette.Midlight, brush3)
        brush4 = QBrush(QColor(104, 120, 118, 255))
        brush4.setStyle(Qt.SolidPattern)
        palette.setBrush(QPalette.Active, QPalette.Dark, brush4)
        brush5 = QBrush(QColor(139, 160, 157, 255))
        brush5.setStyle(Qt.SolidPattern)
        palette.setBrush(QPalette.Active, QPalette.Mid, brush5)
        palette.setBrush(QPalette.Active, QPalette.Text, brush)
        palette.setBrush(QPalette.Active, QPalette.BrightText, brush2)
        palette.setBrush(QPalette.Active, QPalette.ButtonText, brush)
        palette.setBrush(QPalette.Active, QPalette.Base, brush2)
        palette.setBrush(QPalette.Active, QPalette.Window, brush1)
        palette.setBrush(QPalette.Active, QPalette.Shadow, brush)
        palette.setBrush(QPalette.Active, QPalette.AlternateBase, brush3)
        brush6 = QBrush(QColor(255, 255, 220, 255))
        brush6.setStyle(Qt.SolidPattern)
        palette.setBrush(QPalette.Active, QPalette.ToolTipBase, brush6)
        palette.setBrush(QPalette.Active, QPalette.ToolTipText, brush)
        brush7 = QBrush(QColor(0, 0, 0, 128))
        brush7.setStyle(Qt.SolidPattern)
#if QT_VERSION >= QT_VERSION_CHECK(5, 12, 0)
        palette.setBrush(QPalette.Active, QPalette.PlaceholderText, brush7)
#endif
        palette.setBrush(QPalette.Inactive, QPalette.WindowText, brush)
        palette.setBrush(QPalette.Inactive, QPalette.Button, brush1)
        palette.setBrush(QPalette.Inactive, QPalette.Light, brush2)
        palette.setBrush(QPalette.Inactive, QPalette.Midlight, brush3)
        palette.setBrush(QPalette.Inactive, QPalette.Dark, brush4)
        palette.setBrush(QPalette.Inactive, QPalette.Mid, brush5)
        palette.setBrush(QPalette.Inactive, QPalette.Text, brush)
        palette.setBrush(QPalette.Inactive, QPalette.BrightText, brush2)
        palette.setBrush(QPalette.Inactive, QPalette.ButtonText, brush)
        palette.setBrush(QPalette.Inactive, QPalette.Base, brush2)
        palette.setBrush(QPalette.Inactive, QPalette.Window, brush1)
        palette.setBrush(QPalette.Inactive, QPalette.Shadow, brush)
        palette.setBrush(QPalette.Inactive, QPalette.AlternateBase, brush3)
        palette.setBrush(QPalette.Inactive, QPalette.ToolTipBase, brush6)
        palette.setBrush(QPalette.Inactive, QPalette.ToolTipText, brush)
#if QT_VERSION >= QT_VERSION_CHECK(5, 12, 0)
        palette.setBrush(QPalette.Inactive, QPalette.PlaceholderText, brush7)
#endif
        palette.setBrush(QPalette.Disabled, QPalette.WindowText, brush4)
        palette.setBrush(QPalette.Disabled, QPalette.Button, brush1)
        palette.setBrush(QPalette.Disabled, QPalette.Light, brush2)
        palette.setBrush(QPalette.Disabled, QPalette.Midlight, brush3)
        palette.setBrush(QPalette.Disabled, QPalette.Dark, brush4)
        palette.setBrush(QPalette.Disabled, QPalette.Mid, brush5)
        palette.setBrush(QPalette.Disabled, QPalette.Text, brush4)
        palette.setBrush(QPalette.Disabled, QPalette.BrightText, brush2)
        palette.setBrush(QPalette.Disabled, QPalette.ButtonText, brush4)
        palette.setBrush(QPalette.Disabled, QPalette.Base, brush1)
        palette.setBrush(QPalette.Disabled, QPalette.Window, brush1)
        palette.setBrush(QPalette.Disabled, QPalette.Shadow, brush)
        palette.setBrush(QPalette.Disabled, QPalette.AlternateBase, brush1)
        palette.setBrush(QPalette.Disabled, QPalette.ToolTipBase, brush6)
        palette.setBrush(QPalette.Disabled, QPalette.ToolTipText, brush)
#if QT_VERSION >= QT_VERSION_CHECK(5, 12, 0)
        palette.setBrush(QPalette.Disabled, QPalette.PlaceholderText, brush7)
#endif
        SimulationWidget.setPalette(palette)
        font = QFont()
        font.setFamilies([u"Segoe UI"])
        font.setPointSize(12)
        SimulationWidget.setFont(font)
        self.frame = QFrame(SimulationWidget)
        self.frame.setObjectName(u"frame")
        self.frame.setGeometry(QRect(10, 10, 580, 711))
        self.frame.setFocusPolicy(Qt.NoFocus)
        self.frame.setFrameShape(QFrame.StyledPanel)
        self.frame.setFrameShadow(QFrame.Raised)
        self.progressBar = QProgressBar(self.frame)
        self.progressBar.setObjectName(u"progressBar")
        self.progressBar.setGeometry(QRect(10, 520, 561, 31))
        self.progressBar.setValue(0)
        self.verticalLayoutWidget = QWidget(self.frame)
        self.verticalLayoutWidget.setObjectName(u"verticalLayoutWidget")
        self.verticalLayoutWidget.setGeometry(QRect(10, 560, 561, 141))
        self.verticalLayout = QVBoxLayout(self.verticalLayoutWidget)
        self.verticalLayout.setObjectName(u"verticalLayout")
        self.verticalLayout.setContentsMargins(0, 0, 0, 0)
        self.pushButton_comenzar = QPushButton(self.verticalLayoutWidget)
        self.pushButton_comenzar.setObjectName(u"pushButton_comenzar")

        self.verticalLayout.addWidget(self.pushButton_comenzar)

        self.pushButton_detener = QPushButton(self.verticalLayoutWidget)
        self.pushButton_detener.setObjectName(u"pushButton_detener")

        self.verticalLayout.addWidget(self.pushButton_detener)

        self.pushButton_salir = QPushButton(self.verticalLayoutWidget)
        self.pushButton_salir.setObjectName(u"pushButton_salir")

        self.verticalLayout.addWidget(self.pushButton_salir)

        self.pushButton_cargar = QPushButton(self.frame)
        self.pushButton_cargar.setObjectName(u"pushButton_cargar")
        self.pushButton_cargar.setGeometry(QRect(10, 10, 561, 37))
        self.columnView_diagnosticos = QColumnView(self.frame)
        self.columnView_diagnosticos.setObjectName(u"columnView_diagnosticos")
        self.columnView_diagnosticos.setGeometry(QRect(10, 100, 561, 261))
        self.lineEdit_ruta_datos = QLineEdit(self.frame)
        self.lineEdit_ruta_datos.setObjectName(u"lineEdit_ruta_datos")
        self.lineEdit_ruta_datos.setGeometry(QRect(10, 60, 561, 22))
        font1 = QFont()
        font1.setFamilies([u"Segoe UI"])
        font1.setPointSize(8)
        self.lineEdit_ruta_datos.setFont(font1)
        self.listView_simulacion = QListView(self.frame)
        self.listView_simulacion.setObjectName(u"listView_simulacion")
        self.listView_simulacion.setGeometry(QRect(10, 370, 561, 141))
        self.listView_simulacion.setFont(font1)

        self.retranslateUi(SimulationWidget)

        QMetaObject.connectSlotsByName(SimulationWidget)
    # setupUi

    def retranslateUi(self, SimulationWidget):
        SimulationWidget.setWindowTitle(QCoreApplication.translate("SimulationWidget", u"Simulaci\u00f3n", None))
        self.pushButton_comenzar.setText(QCoreApplication.translate("SimulationWidget", u"Comenzar Simulaci\u00f3n", None))
        self.pushButton_detener.setText(QCoreApplication.translate("SimulationWidget", u"Detener Simulaci\u00f3n", None))
        self.pushButton_salir.setText(QCoreApplication.translate("SimulationWidget", u"Salir", None))
        self.pushButton_cargar.setText(QCoreApplication.translate("SimulationWidget", u"Cargar datos", None))
    # retranslateUi

