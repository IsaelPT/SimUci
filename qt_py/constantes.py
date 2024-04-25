class Rutas:
    """Para variables con rutas estáticas para cargar componentes necesarios."""

    SIMULATIONWIDGET_UI = "qt_ui/simulation_widget.ui"
    MAINWINDOW_UI = "qt_ui/main_window.ui"


class Estilos:
    """Contiene estilos personalizados que se utilizan en botones y otros componentes en las interfaces gráficas."""

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


class Colores:
    VERDE_CLARO = "#b2f2bb"
    GRIS = "#8f8f91"
