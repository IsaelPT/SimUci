import traceback
import typing

from PyQt5.QtWidgets import (
    QWidget,
    QFileDialog,
    QMessageBox,
    QTableWidgetItem,
    QMainWindow,
)
from PyQt5.uic import loadUi
from PyQt5.QtGui import QIcon

from qt_py.resources import Rutas, Estilos, CustomQMessageBox

from uci import procesar_datos as proc_d
from uci.uci_simulacion import Uci


class SimulationWindow(QWidget):
    """
    Ventana donde se llevan a cabo simulaciones con datos brindados a través de un archivo de datos `.csv`.
    Se muestran los datos por pantalla.
    """

    ruta_archivo_csv: str = None
    FILAS: int = 0
    """FILAS contiene la cantidad de filas de diagnosticos presentes en la tabla.
    Se inicia su valor al iniciar la tabla."""

    def __init__(self, main_win) -> None:
        super().__init__()

        # Referencia a la clase padre (MainWindow).
        self.main_win: QMainWindow = main_win

        # baseinstance: SimulationWindow
        loadUi(Rutas.Ui_Files.SIMULATIONWIDGET_UI, self)

        # self.threads: list[Uci] = []  # Hilos de esta ventana.
        self.runner: Uci = None

        # Conexiones de los componentes.
        self.pB_cargar.clicked.connect(self.cargar_csv)
        self.pB_comenzar.clicked.connect(self.comenzar_simulacion)
        self.pB_detener.clicked.connect(self.detener_simulacion)
        self.pB_salir.clicked.connect(self.cerrar_ventana)

        # Estilos personalizados a los componentes.
        self.pB_cargar.setStyleSheet(Estilos.botones["botones_acciones_verdes"])
        self.pB_comenzar.setStyleSheet(Estilos.botones["botones_acciones_verdes"])
        self.pB_detener.setStyleSheet(Estilos.botones["botones_acciones_verdes"])
        self.pB_salir.setStyleSheet(Estilos.botones["botones_acciones_verdes"])

        # Ajustes iniciales a componentes.
        self.pB_detener.setEnabled(False)
        self.setWindowIcon(QIcon(Rutas.Iconos.WINDOWICON_SIMULATION))

        # Otros componentes.
        self.lineEdit_ruta_datos.setText("Ruta de archivo...")

    def cargar_csv(self) -> None:
        """
        Carga un archivo `.csv` a la aplicación a través de un `QFileDialog`.
        """

        self.ruta_archivo_csv, _ = QFileDialog.getOpenFileName(
            self, "Abrir CSV", "", "Archivos CSV (*.csv)"
        )

        if self.ruta_archivo_csv == "":
            return

        try:
            diagnosticos = proc_d.get_diagnostico_list(self.ruta_archivo_csv)
            self._init_tabla_diagnosticos(diagnosticos)  # Ingresar datos en Tabla.
            self.lineEdit_ruta_datos.setText(self.ruta_archivo_csv)
            print(f"Archivo CSV '{self.ruta_archivo_csv}' cargado correctamente!")
        except:
            print(f"Ocurrió un error al cargar el archivo:\n{traceback.format_exc()}")

    def comenzar_simulacion(self) -> None:
        """
        Da inicio a la simulación al ser pulsado el botón de "Comenzar Simulación".

        Funcionamiento
        --------------

        1- Se comprueba que se haya especificado una ruta donde se encuentra el achivo de datos `.csv`.
        En caso de no haberse especificado, mostrar un mensaje de Advertencia y devolver `None`.

        2- Se almacenan los datos de la ruta, se cargan y almacenan los diagnosticos y los porcientos
        del archivo de datos `.csv`. De ocurrir alguna particularidad al respecto, se muestra por pantalla
        un traceback del error.

        3- Se instancia la clase `Uci` y se le pasan los datos de la ruta, los diagnosticos y los porcientos.

        4- Se hacen las conexiones necesarias con la UI y se desactivan botones para comenzar la simulación.

        5- Se inicia la simulación. Cuando finaliza la simulación se debe mostrar un mensaje informativo del proceso.
        """

        if self.ruta_archivo_csv is None:
            QMessageBox.warning(
                self,
                "Imposible iniciar simulación",
                "No se puede iniciar la simulación debido a que no hay datos para simular. Por favor, cargue los datos.",
            )
            return
        try:
            print("-- Se presionó el botón de 'Comenzar Simulacion' --")

            ruta = self.ruta_archivo_csv
            diagnosticos_tabla = proc_d.get_diagnostico_list(self.ruta_archivo_csv)
            porcientos_tabla = self._get_porcientos_de_tabla()

            if ruta is None or diagnosticos_tabla is None or porcientos_tabla is None:
                raise RuntimeError("Hubo un error al momento de iniciar la simulacion.")

            self.runner = Uci(ruta, diagnosticos_tabla, porcientos_tabla)

            self.runner.signal.signal_progBarr.connect(self._update_progressBarr)
            self.runner.signal.signal_tiempo.connect(self._show_mensaje_finalizado)
            self.runner.signal.signal_terminated.connect(self.__switch)
            self.runner.signal.signal_interruption.connect(
                self._show_mensaje_interrupcion
            )

            self.runner.start()

            # Para tener constancia de los hilos que están activos.
            print(f"{self.runner.name} esta corriendo: {self.runner.is_alive()}")
        except:
            print(f"Error a la hora de correr la simulación:\n{traceback.format_exc()}")

    def detener_simulacion(self) -> None:
        """Detiene la simulación a medio proceso.

        Args:
            show_warning_msg (bool): Útil para establecer si se mostrará un mensaje de advertencia o no tras cancelar la simulación. True mostrará el mensaje, en caso contrario, no.
        """

        try:
            print("-- Se presionó el botón de 'Detener Simulación' --")
            self.runner.stop()
            self.progressBar.setValue(0)
            self.pB_cargar.setEnabled(True)
            self.pB_comenzar.setEnabled(True)
            self.pB_detener.setEnabled(False)
        except:
            print(f"Error deteniendo la simulación:\n{traceback.format_exc()}")

    def cerrar_ventana(self) -> None:
        """Cierra esta ventana de Simulación."""

        try:
            if self.runner is not None and self.runner.is_alive():
                self.detener_simulacion()
                self._show_mensaje_interrupcion()
            self.close()
        except:
            print(f"Ocurrió un error al cerrar la ventana:\n{traceback.format_exc()}")

    def _init_tabla_diagnosticos(self, diagnosticos) -> None:
        """Inicializa el widget de la tabla de diagnosticos.
        También inicia la columna de porcientos con valores de 0.


        Args:
            diagnosticos (list): Lista con los diagnosticos extraidos del archivo de datos `.csv`.
        """

        self.FILAS = len(diagnosticos)

        print(f"Cantidad de diagnosticos: {self.FILAS}")
        print(f"Lista de diagnosticos importada:\n{self.FILAS}")

        self.tableWidget.setRowCount(self.FILAS)

        for i in range(self.FILAS):
            self.tableWidget.setItem(i, 0, QTableWidgetItem(diagnosticos[i]))
            self.tableWidget.setItem(i, 1, QTableWidgetItem("0"))

    def _update_progressBarr(self, contador: int) -> None:
        """Actualiza el progreso de la barra de simulación.


        Args:
            contador (int): Contador que se utiliza para actualizar el progreso de la barra de simulación.
        """

        self.progressBar.setValue(int(contador / 17880 * 100))

    def _get_porcientos_de_tabla(self) -> list[float] | None:
        """Obtiene del QTableWidget los porcentajes que actualmente se han ingresado y se validan.
        En caso de que haya un porcentaje incorrecto, se muestra por pantalla un mensaje de advertencia
        con instrucciones al usuario para corregir los errores.


        Returns:
            list[float] | None: Lista con los porcentajes extraidos de la tabla o None en caso de error.
        """

        porcentajes = []
        incorrectos = []  # Para mantener seguimiento de las filas incorrectas.
        for index in range(self.FILAS):
            item_porcentaje: QTableWidgetItem = self.tableWidget.item(index, 1)
            porcentaje = (
                item_porcentaje.text()
                .replace(" ", "")
                .replace(",", ".")
                .replace("%", "")
            )
            # Comprobar que es decimal y de punto flotante.
            if not self._is_floatValid(porcentaje):
                incorrectos.append(index + 1)
            else:
                try:
                    parsed_item = float(porcentaje)  # Convertir a float para guardarlo.
                    parsed_item = round(parsed_item, 2)  # Disminuir nivel de precisión.
                except:
                    print(f"Error a la hora de parsear:\n{traceback.format_exc()}")
                    return

                # Finalmente se comprueba que el porcentaje está en rango para agregarlo a la lista de porcientos.
                (
                    porcentajes.append(parsed_item)
                    if parsed_item >= 0 and parsed_item <= 100
                    else incorrectos.append(index + 1)
                )

        if len(incorrectos) == 1:
            title = "Porciento incorrecto"
            msg = f"Se ha encontrado que en la columna de porcentajes, precisamente en la fila {incorrectos[0]}, un porciento ha sido ingresado incorrectamente. Por favor, rectifique para poder iniciar la simulación."
            QMessageBox.warning(self, title, msg)
            return None
        if len(incorrectos) > 1:
            title = "Porcientos incorrectos"
            msg = f"Se han encontrado que en la columna de porcentajes, precisamente en las filas {incorrectos}, porcientos han sido ingresado incorrectamente. Por favor, rectifique para poder iniciar la simulación."
            QMessageBox.warning(self, title, msg)
            return None

        print(f"Lista de porcentajes:\n{porcentajes}")
        return porcentajes

    def _is_floatValid(self, cadena: str) -> bool:
        """Comprueba que la cadena sea un número decimal de punto flotante. Este hecho está dado cuando el número
        es una cadena de caracteres, tiene un caracter de `.` que se para la parte decimal de la fraccionaria
        y además que no se encuentre otro tipo de caracter que no sean números en la cadena.


        Args:
            cadena (str): Cadena a evaluar.

        Returns:
            bool: `True` si la cadena tiene la estructura de un número decimal con punto flotante. `False` en caso contrario.
        """

        if cadena.isdecimal():
            return True
        if "." in cadena and cadena.count(".") == 1:
            _ = cadena.split(".")
            if _[0].isdecimal() and _[1].isdecimal():
                return True
        return False

    def _show_mensaje_finalizado(self, tiempo: float) -> None:
        """Muestra un mensaje de que la simulación ha finalizado y se muestra además el tiempo que duró esta simulación.

        Args:
            tiempo (float): Tiempo transcurrido en la simulación.
        """

        CustomQMessageBox.finalizado(
            self,
            "Simulación finalizada",
            f"La simulación terminó a los {tiempo} segundos.",
        )

    def _show_mensaje_interrupcion(self) -> None:
        """Muestra un mensaje de que se ha interrumpido la simulación"""

        CustomQMessageBox.interrupcion(
            self,
            "Simulación interrumpida",
            "La simulación ha sido interrumpida por el usuario.",
        )

    def __switch(self, trigger: bool):
        """Propósito de la función es cambiar la visualización de los botones
        atendiendo a una señal booleana que se emite en el proceso de la simulación.

        Args:
            trigger (bool): Si es `True` fue que se terminó la simulación y los botones se activarán de nuevo, caso contrario es que la simulación está en proceso.
        """
        if trigger:
            self.pB_comenzar.setEnabled(True)
            self.pB_cargar.setEnabled(True)
            self.pB_detener.setEnabled(False)
        else:
            self.pB_comenzar.setEnabled(False)
            self.pB_cargar.setEnabled(False)
            self.pB_detener.setEnabled(True)
