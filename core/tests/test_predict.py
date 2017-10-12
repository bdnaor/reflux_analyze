from unittest import TestCase

from core.cnn_manager import CNNManager


class TestPredictFrame(TestCase):
    def test_predict(self):
        self.cnn = CNNManager().models[0]
        frame, label = self.cnn.get_random_prediction()
        self.assertTrue(self.cnn.predict(frame) in [0, 1])
