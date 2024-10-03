import unittest

from constants import *  # Asegúrate de importar tus constantes
from helpers import *


class MyTestCase(unittest.TestCase):

    def test_key_categ(self):
        # Prueba con categorías válidas
        self.assertEqual(key_categ("va", "Tubo endotraqueal"), 0)
        self.assertEqual(key_categ("diag", "Intoxicación exógena"), 1)
        self.assertEqual(key_categ("insuf", "Respiratorias"), 1)

        # Prueba con viceversa
        self.assertEqual(key_categ("va", 0, viceversa=True), "Tubo endotraqueal")

        # Prueba con categoría inválida
        with self.assertRaises(Exception):
            key_categ("invalid", "value")

        # Prueba con valor no encontrado
        with self.assertRaises(Exception):
            key_categ("va", "Non-existent value")

    def test_value_is_zero(self):
        # Pruebas con diferentes entradas
        self.assertTrue(value_is_zero(0))
        self.assertTrue(value_is_zero("vacío"))
        self.assertTrue(value_is_zero([0, "vacío"]))
        self.assertFalse(value_is_zero(5))
        self.assertFalse(value_is_zero(["vacío", 1]))

        # Prueba con entrada inválida
        with self.assertRaises(ValueError):
            value_is_zero(None)

    def test_generate_id(self):
        # Generar ID de longitud 10
        self.assertEqual(len(generate_id()), 10)

        # Generar ID de longitud 0 debe lanzar excepción
        with self.assertRaises(Exception):
            generate_id(0)

    def test_to_hours(self):
        # Probar conversión de días a horas
        self.assertEqual(to_hours(3), 72)
        self.assertEqual(to_hours(1), 24)
        self.assertEqual(to_hours(0), 0)

    def test_value_is_zero_with_edge_cases(self):
        # Probar valores límite para 'value_is_zero'
        self.assertTrue(value_is_zero([0] * (EDAD_MAX - EDAD_MIN + 1)))  # Lista de ceros del tamaño del rango de edad


if __name__ == '__main__':
    unittest.main()
