from core.data_set import DataSet
from utils.singleton import singleton


@singleton
class DataSetManager(object):
    def __init__(self):
        self.data_sets = []

    def _get_or_create_data_set(self, img_rows, img_cols):
        for _set in self.data_sets:
            if _set.img_rows == img_rows and _set.img_cols == img_cols:
                return _set
        _data_set = DataSet(img_rows, img_cols)
        self.data_sets.append(_data_set)
        return _data_set

    def get_data_set_split_frames(self, img_rows, img_cols, train_ratio):
        _set = self._get_or_create_data_set(img_rows, img_cols)
        return _set.get_data_set_split_frames(train_ratio)

    def get_data_set_split_cases(self, img_rows, img_cols, train_ratio):
        _set = self._get_or_create_data_set(img_rows, img_cols)
        return _set.get_data_set_split_cases(train_ratio)

    def get_categories(self, img_rows, img_cols):
        _set = self._get_or_create_data_set(img_rows, img_cols)
        return _set.categories

    def get_adaptation_dataset(self, img_rows, img_cols):
        _set = self._get_or_create_data_set(img_rows, img_cols)
        return _set.adaptation_dataset