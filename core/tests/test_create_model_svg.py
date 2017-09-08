from unittest import TestCase

from core.cnn import CNN


class TestCreateModelSVG(TestCase):
    def test_create_model_svg(self):
        cnn = CNN({'model_name': 'naor_first_model'}, True)
        cnn.create_model_svg()