from unittest import TestCase

import os

from core.cnn import CNN
from manage import ROOT_DIR


class TestSaveLoadCNN(TestCase):
    def setUp(self):
        self.path = os.path.join(ROOT_DIR, 'cnn_models', 'test_model')
        self.cnn = CNN(nb_epoch=15, pool_size=3, kernel_size=4)
        self.cnn.build_model()
        self.cnn.save(self.path)
        self.cnn = None

    def test_load_cnn(self):
        self.cnn = CNN()
        self.cnn.load(self.path)
        self.assertEqual(self.cnn.nb_epoch, 15)
        self.assertEqual(self.cnn.pool_size, (3, 3))
        self.assertEqual(self.cnn.kernel_size, 4)

    @classmethod
    def tearDownClass(cls):
        file_list = os.listdir(os.path.join(ROOT_DIR, 'cnn_models'))
        for _file in file_list:
            if 'test' in _file:
                os.remove(os.path.join(ROOT_DIR, 'cnn_models', _file))
