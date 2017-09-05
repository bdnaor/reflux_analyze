from unittest import TestCase

import os

from core.cnn import CNN
from core.cnn_manager import CNNManager
from manage import ROOT_DIR


class TestCNNManager(TestCase):
    def test_add_model(self):
        cnn_manager = CNNManager()
        self.expected_file = os.path.join(ROOT_DIR, 'cnn_models', 'cpu_cnn_model_%s' % (cnn_manager.last_index+1))
        self.test_cnn = CNN(img_rows=300)
        cnn_manager.add_model(self.test_cnn)
        self.assertTrue(os.path.exists(self.expected_file+'.h5'))
        self.assertTrue(os.path.exists(self.expected_file+'.json'))
        cnn_manager.remove_model(self.test_cnn)
        self.assertFalse(os.path.exists(self.expected_file + '.h5'))
        self.assertFalse(os.path.exists(self.expected_file + '.json'))
