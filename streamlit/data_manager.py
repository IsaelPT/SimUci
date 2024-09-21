from utils.constants.categories import TIPO_VENT, DIAG_PREUCI, INSUF_RESP


def key_categ(categoria: str, valor: str | int, viceversa: bool = False) -> int | str:
    """
    Obtiene la llave (key, k) que constituye un valor si está presente en la colección de categorías definidas.
    @param categoria: Las categorías deben ser entre "va", "diag" y "insuf".
    @param valor: Es el valor que se pasa para buscar su llave.
    @param viceversa: Determina si en lugar de buscar el valor, se busca la llave (key).
    @return: Llave que representa en las colecciones de categorías el valor que se pasa por parámetros.
    """
    match categoria:
        case "va":
            categorias = TIPO_VENT
        case "diag":
            categorias = DIAG_PREUCI
        case "insuf":
            categorias = INSUF_RESP
        case _:
            raise Exception(f"La categoría que se selecciona no existe {categoria}.")
    for k, v in categorias.items():
        if not viceversa:
            if v == valor:
                return k
        else:
            if k == valor:
                return v
    if not viceversa:
        raise Exception(f"El valor (value) que se proporcionó no se encuentra en el conjunto de categorías {categoria}")
    else:
        raise Exception(f"La llave (key) que se proporcionó no se encuentra en el conjunto de categórias {categoria}")
