from unittest import TestCase
from core.cnn import CNN


class TestLoadSaveLoadModel(TestCase):
    def test(self):
        def excpected_from_the_cnn(cnn):
            self.assertEqual(cnn.tp, 396)
            self.assertEqual(cnn.tn, 970)
            self.assertEqual(cnn.fp, 6)
            self.assertEqual(cnn.fn, 61)
        cnn = CNN({'model_name': 'naor_first_model'}, True)
        cnn._calculate_confusion_matrix()
        excpected_from_the_cnn(cnn)
        cnn.save()
        cnn = CNN({'model_name': 'naor_first_model'}, True)
        excpected_from_the_cnn(cnn)
