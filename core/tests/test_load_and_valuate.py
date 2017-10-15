from unittest import TestCase
import os
from core.cnn import CNN


class TestLoadAndValuate(TestCase):
    def test_load_evaluate(self):
        cnn = CNN({'model_name': 'model1'}, _reload=True)
        cnn.load_data_set()
        cnn._calculate_confusion_matrix()
