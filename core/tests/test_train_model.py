from unittest import TestCase

from core.cnn import CNN


class TestTrainModel(TestCase):
    def test_train_model(self):
        # cnn = CNN({'model_name': 'split_cases_50_50_2_layer', 'img_rows': 200, 'img_cols': 200})
        cnn = CNN({'model_name': 'model_with_gabor'}, True)
        # cnn = CNN({'model_name': 'split_cases_dropout_50', 'img_rows': 200, 'img_cols': 200})
        # cnn = CNN({'model_name': 'test_split_case'}, True)
        # cnn = CNN({'model_name': 'test_split_case', 'img_rows': 200, 'img_cols': 200})
        # cnn = CNN({'model_name': 'test_split_case'}, True)
        # cnn = CNN({'model_name': 'second_model'}, True)
        # cnn = CNN({'model_name': 'naor_first_model'}, True)
        # cnn._calculate_confusion_matrix()
        # self.assertEqual(cnn.tp, 396)
        # self.assertEqual(cnn.tn, 970)
        # self.assertEqual(cnn.fp, 6)
        # self.assertEqual(cnn.fn, 61)
        cnn.train_model(n_epoch=4)

        # Reload CNN
        # cnn = CNN({'model_name': 'naor_first_model'}, True)
        # cnn._calculate_confusion_matrix()
        # self.assertNotEqual(cnn.tp, 396)
        # self.assertNotEqual(cnn.tn, 970)
        # self.assertNotEqual(cnn.fp, 6)
        # self.assertNotEqual(cnn.fn, 61)



