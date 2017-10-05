from unittest import TestCase
from core.cnn import CNN


class TestLoadSaveLoadModel(TestCase):
    def test(self):
        cnn = CNN({'model_name': 'model7(new_aug)'}, True)
        cnn.load_datasets()
        cnn._calculate_confusion_matrix()
