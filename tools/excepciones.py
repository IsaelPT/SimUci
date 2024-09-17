import traceback


class ExceptionSaver:
    """Permite guardar en un archivo `.log` las excepciones y errores que van ocurriendo."""
    __PATH = "tools/errorslog/errors.log"

    def save(self, excepcion: Exception) -> None:
        """
        Guarda en un archivo `.log` la excepción que se pasa por parámetros, acompañada del tiempo de ocurrencia del
        mismo y la explicación de esta excepción.

        Args:
            excepcion (Exception): La excepción a guardar.
        """
        if isinstance(excepcion, Exception):
            with open(self.__PATH, 'a', encoding="UTF-8") as error_file:
                msg = (
                    f"[{self.__get_time__()}] -> [Excepción]: {excepcion}\nDetalles de la excepción"
                    f":\n{traceback.format_exc()}{'-' * 132}\n"
                )
                error_file.write(msg)
        else:
            print(f"No se pudo archivar la excepción:\n{self.__str__()} no es una excepción! ({type(Exception)})")

    @staticmethod
    def __get_time__() -> str:
        """
        Obtiene el tiempo y fecha actual.
        Returns:
            str: Cadena con formato: dd/mm/yy - hh/mm/ss
        """
        import time
        current_time = time.localtime()
        year, month, day = current_time[:3]
        hours, minutes, seconds = current_time[3:6]
        return f"{day}/{month}/{year} - {hours}:{minutes}:{seconds}"
