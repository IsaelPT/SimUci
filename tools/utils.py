class Rutas:
    """Para variables con rutas estáticas para cargar componentes necesarios."""

    class UiFiles:
        """Donde se localizan los archivos `.ui` que sirven para construir las GUI."""

        MAINWINDOW_UI = "ui_templates/main_window.ui"
        SIMULATIONWIDGET_UI = "ui_templates/simulation_widget.ui"
        GESTIONPACIENTEWIDGET_UI = "ui_templates/gestion_pacientes.ui"

    class Iconos:
        """Donde se localizan los archivos de iconos."""

        WINDOWICON_HEALTH = "ui_utils/icons/MageHealthSquareFill.png"
        WINDOWICON_SIMULATION = "ui_utils/icons/FluentMdl2TestPlan.png"
        ICON_QMESSAGEBOX_OK = "ui_utils/icons/FlatColorIconsOk.png"
        ICON_QMESSAGEBOX_STOP = "ui_utils/icons/MaterialSymbolsStopCircleOutlineRounded.png"
        ICON_QMESSAGEBOX_SIMULATIONBREAK = "ui_utils/icons/CodiconBeakerStop.png"
        ICON_QMESSAGEBOX_TIME = "ui_utils/icons/PhTimerDuotone.png"

    class Data:
        """Donde se localizan los archivos de datos que utiliza la aplicación."""

        DATOS_HISTORIAL_PACIENTES = "data/historial.json"


class VariablesConstantes:
    """Donde se contiene variables constantes para la aplicación"""

    DIAG_PREUCI = {
        0: "Vacío", 1: "Intoxicación exógena", 2: "Coma", 3: "Trauma craneoencefálico severo",
        4: "SPO de toracotomía", 5: "SPO de laparotomía", 6: "SPO de amputación",
        7: 'SPO de neurología', 8: 'PCR recuperado', 9: 'Encefalopatía metabólica', 10: 'Encefalopatía hipóxica',
        11: 'Ahorcamiento incompleto', 12: 'Insuficiencia cardiaca descompensada', 13: 'Obstétrica grave',
        14: 'EPOC descompensada', 15: 'ARDS', 16: 'BNB-EH', 17: 'BNB-IH', 18: 'BNV', 19: 'Miocarditis',
        20: 'Leptospirosis', 21: 'Sepsis grave', 22: 'DMO', 23: 'Shock séptico', 24: 'Shock hipovolémico',
        25: 'Shock cardiogénico', 26: 'IMA', 27: 'Politraumatizado', 28: 'Crisis miasténica',
        29: 'Emergencia hipertensiva', 30: 'Status asmático', 31: 'Status epiléptico', 32: 'Pancreatitis',
        33: 'Embolismo graso', 34: 'Accidente cerebrovascular', 35: 'Síndrome de apnea del sueño',
        36: 'Sangramiento digestivo', 37: 'Insuficiencia renal crónica', 38: 'Insuficiencia renal aguda',
        39: 'Trasplante renal', 40: 'Guillain Barré'
    }
    """Diccionario con par `int`:`str` (key, diagnóstico) que contiene todos los diagnosticos preUCI. Estos diagnósticos
    son importantes para la simulación."""

    TIPO_VENT = {
        0: "Tubo endotraqueal",
        1: "Traqueostomía",
        2: "Ambas"
    }
    """Tipo de ventilación artificial mecánica."""
