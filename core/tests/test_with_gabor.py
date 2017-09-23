from unittest import TestCase
import os
from core.cnn import CNN


class TestWithGabor(TestCase):
    def test_train_model(self):
        _con_mat = [[25, 25, 25, 25], [30, 20, 30, 20], [50, 0, 0, 50]]
        cnn = CNN({'model_name': 'kernal_5X5', 'img_rows': 75, 'img_cols': 75, 'kernel_size': (5, 5)})
        cnn.con_mat_train = _con_mat
        cnn.con_mat_val = _con_mat
        cnn._save_only_best()
        self.assertTrue(os.path.exists(os.path.join(cnn.model_path + '.h5(weights)')))
        self.assertTrue(os.path.exists(os.path.join(cnn.model_path+'.json')))
        del cnn
        cnn = CNN({'model_name': 'kernal_5X5'}, True)
        self.assertEqual(_con_mat, cnn.con_mat_train)
        self.assertEqual(_con_mat, cnn.con_mat_val)



