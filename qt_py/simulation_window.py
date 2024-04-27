import traceback
from typing import List

from PyQt5.QtWidgets import QWidget, QFileDialog, QMessageBox, QTableWidgetItem
from PyQt5.uic import loadUi

from qt_py.constantes import Rutas, Estilos

from uci import procesar_datos as proc_d
from uci.uci_simulacion import Uci


class SimulationWindow(QWidget):
    """
    Ventana donde se llevan a cabo simulaciones con datos brindados a través de un archivo de datos `.csv`.
    Se muestran los datos por pantalla.
    """

    ruta_archivo_csv: str = None

    def __init__(self, main_win) -> None:
        super().__init__()
        self.main_win = main_win  # Referencia a la clase padre (MainWindow).
        loadUi(Rutas.SIMULATIONWIDGET_UI, self)  # baseinstance: SimulationWindow

        self.threads = []  # Hilos de esta ventana.

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

        # Ajustes iniciales a botones.
        self.pB_detener.setEnabled(False)

        # Otros componentes.
        self.lineEdit_ruta_datos.setText("Ruta de archivo...")

    def cargar_csv(self) -> None:
        """
        Carga un archivo `.csv` a la aplicación a través de un `QFileDialog`.
        """

        self.ruta_archivo_csv, _ = QFileDialog.getOpenFileName(
            self, "Abrir CSV", "", "Archivos CSV (*.csv)"
        )

        if self.ruta_archivo_csv is not None:
            try:
                diagnosticos = proc_d.get_diagnostico_list(self.ruta_archivo_csv)
                self._init_tabla_diagnosticos(diagnosticos)  # Ingresar datos en Tabla.
                self.lineEdit_ruta_datos.setText(self.ruta_archivo_csv)
                print(f"Archivo CSV '{self.ruta_archivo_csv}' cargado correctamente!")
            except:
                print(
                    f"Ocurrió un error al cargar el archivo:\n{traceback.format_exc()}"
                )
        else:
            raise Exception("La ruta de archivos no existe u ocurrió un error.")

    def comenzar_simulacion(self) -> None:
        """
        Da inicio a la simulación al ser pulsado el botón de "Comenzar Simulación".

        Funcionamiento
        -------------

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
            if (
                ruta is not None
                and diagnosticos_tabla is not None
                and porcientos_tabla is not None
            ):
                runner = Uci(ruta, diagnosticos_tabla, porcientos_tabla)
                runner.signal.signal_progBarr.connect(self._update_progressBarr)
                runner.signal.signal_terminated.connect(self.pB_comenzar.setEnabled)
                runner.signal.signal_tiempo.connect(self._show_mensaje_finalizado)
                self.pB_comenzar.setEnabled(False)
                self.pB_cargar.setEnabled(False)
                self.pB_detener.setEnabled(True)
                runner.start()
                self.threads.append(runner)
        except:
            print(
                f"Ocurrió un error inesperado a la hora de correr la simulación:\n{traceback.format_exc()}"
            )

    def detener_simulacion(self, show_warning_message: bool) -> None:
        """Detiene la simulación a medio proceso.

        Args:
            show_warning_message (bool): Útil para determinar si se mostrará un mensaje de advertencia o no.
        """

        try:
            print("Se presionó el botón de 'Detener Simulación'.")
            for runner in self.threads:
                runner.stop()
            self.progressBar.setValue(0)
            self.pB_cargar.setEnabled(True)
            self.pB_comenzar.setEnabled(True)
            self.pB_detener.setEnabled(False)
            if not show_warning_message:
                QMessageBox().warning(
                    self, "Detención de simulación", "Se ha detenido la simulación."
                )
        except:
            print(f"Ocurrió un error inesperado:\n{traceback.format_exc()}")

    def cerrar_ventana(self) -> None:
        """Cierra esta ventana de Simulación."""

        try:
            self.detener_simulacion(False)
            self.close()
        except:
            print(f"Ocurrió un error al cerrar la ventana:\n{traceback.format_exc()}")

    def _init_tabla_diagnosticos(self, diagnosticos) -> None:
        """Inicializa el widget de la tabla de diagnosticos.
        También inicia la columna de porcientos con valores de 0.


        Args:
            diagnosticos (_type_): Lista con los diagnosticos extraidos del archivo de datos `.csv`.
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

    def _get_porcientos_de_tabla(self) -> List[float]:
        """Obtiene del QTableWidget los porcentajes que actualmente se han ingresado y los valida.
        En caso de que haya un porcentaje incorrecto, se muestra por pantalla un mensaje de advertencia
        con instrucciones al usuario para corregir los errores.

        Returns:
            List[float]: Lista con los porcentajes extraidos de la tabla.
        """

        # TODO: Permitir que se validen correctamente los valores con coma flotantes presentes en la tabla.

        porcentajes = []
        incorrectos = []  # Para mantener seguimiento de las filas incorrectas.
        for index in range(self.FILAS):
            item_porcentaje = self.tableWidget.item(index, 1)
            porcentaje: str = item_porcentaje.text()
            if porcentaje.isdigit():
                parsed_item = float(item_porcentaje.text())
                if parsed_item >= 0 and parsed_item <= 100:
                    porcentajes.append(parsed_item)
                else:
                    incorrectos.append(index + 1)
            else:
                incorrectos.append(index + 1)

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

    def _show_mensaje_finalizado(self, tiempo: float):
        """Muestra un mensaje de que la simulación ha finalizado y se muestra además el tiempo que duró esta simulación.

        Args:
            tiempo (float): Tiempo transcurrido en la simulación.
        """

        QMessageBox.warning(
            self,
            "Simulación finalizada",
            f"La simulación terminó a los {tiempo} segundos.",
        )
