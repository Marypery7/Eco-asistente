import unittest

from app import detectar_categoria_o_punto_limpio


class TestClasificacionResiduo(unittest.TestCase):
    def test_producto_no_conocido_muestra_punto_limpio(self):
        resultado = detectar_categoria_o_punto_limpio("laptop dañado")
        self.assertEqual(resultado, "Ir al punto limpio")

    def test_producto_conocido_retorna_categoria(self):
        resultado = detectar_categoria_o_punto_limpio("botella de vidrio")
        self.assertEqual(resultado, "vidrio")

    def test_producto_exactamente_en_el_csv_retorna_su_categoria(self):
        resultado = detectar_categoria_o_punto_limpio("botella cerveza")
        self.assertEqual(resultado, "vidrio")


if __name__ == "__main__":
    unittest.main()
