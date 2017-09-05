from unittest import TestCase

from core.cnn import CNN


class TestTrainModel(TestCase):
    def test_train_model(self):
        self.test_cnn = CNN(nb_epoch=2)
        self.test_cnn.train_model()

