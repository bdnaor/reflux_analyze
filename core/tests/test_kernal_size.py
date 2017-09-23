from unittest import TestCase
import os
from core.cnn import CNN


class TestWithGabor(TestCase):
    def test_train_model(self):
        _con_mat = [[25, 25, 25, 25], [30, 20, 30, 20], [50, 0, 0, 50]]
        model_name = 'kernal_6X6'
        cnn = CNN({'model_name': model_name, 'img_rows': 75, 'img_cols': 75, 'kernel_size': (8, 8)})
        cnn.con_mat_train = _con_mat
        cnn.con_mat_val = _con_mat
        cnn._save_only_best()
        self.assertTrue(os.path.exists(os.path.join(cnn.model_path + '.h5(weights)')))
        self.assertTrue(os.path.exists(os.path.join(cnn.model_path+'.json')))
        cnn.train_model(1)
        del cnn
        cnn = CNN({'model_name': model_name}, True)
        self.assertEqual(_con_mat, cnn.con_mat_train)
        self.assertEqual(_con_mat, cnn.con_mat_val)



