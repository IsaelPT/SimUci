from dataclasses import dataclass


@dataclass
class Paciente:
    id: str
    edad: int
    apache: int
    diag1: str
    diag2: str
    diag3: str
    diag4: str
    tiempo_va: int
    tipo_va: str
    estad_uti: int
    estad_preuti: int
