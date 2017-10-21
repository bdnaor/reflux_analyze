from unittest import TestCase
from core.data_set import DataSet


class TestLoadDataSet(TestCase):
    def test_load_data_set(self):
        _data_set = DataSet(50, 50)
        _data_set.get_data_set_split_frames(train_ratio=0.6)
        _data_set.get_data_set_split_cases(train_ratio=0.6)