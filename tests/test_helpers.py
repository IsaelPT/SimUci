import unittest

from st_utils.constants import *
from st_utils.helpers import *


class MyTestCase(unittest.TestCase):
    """Clase para testeo de funciones helpers."""

    def test_key_categ(self):
        self.assertEqual(key_categ("va", "Tubo endotraqueal"), 0)
        self.assertEqual(key_categ("diag", "Intoxicación exógena"), 1)
        self.assertEqual(key_categ("insuf", "Respiratorias"), 1)

        self.assertEqual(key_categ("va", 0, viceversa=True), "Tubo endotraqueal")

        with self.assertRaises(Exception):
            key_categ("invalid", "value")

        with self.assertRaises(Exception):
            key_categ("va", "Non-existent value")

    def test_value_is_zero(self):
        self.assertTrue(value_is_zero(0))
        self.assertTrue(value_is_zero("vacío"))
        self.assertTrue(value_is_zero([0, "vacío"]))
        self.assertFalse(value_is_zero(5))
        self.assertFalse(value_is_zero(["vacío", 1]))

        with self.assertRaises(ValueError):
            value_is_zero(None)

    def test_generate_id(self):
        self.assertEqual(len(generate_id()), 10)

        with self.assertRaises(Exception):
            generate_id(0)

    def test_to_hours(self):
        self.assertEqual(3, 72)
        self.assertEqual(1, 24)
        self.assertEqual(0, 0)

    def test_value_is_zero_with_edge_cases(self):
        self.assertTrue(value_is_zero([0] * (EDAD_MAX - EDAD_MIN + 1)))


if __name__ == '__main__':
    unittest.main()
