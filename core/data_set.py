import os
import gc
import numpy as np
from PIL import Image
from sklearn.utils import shuffle
from sklearn.model_selection import train_test_split
from keras.utils import np_utils

from manage import ROOT_DIR
from utils.prepare_dataset import reshape_images


def _normalized_data_set(X_train, X_test, y_train, y_test, category):
    X_train = X_train.astype('float32')
    X_test = X_test.astype('float32')

    # help for faster convert
    X_train /= 255
    X_test /= 255

    # convert class vector to binary class matrices
    y_train = np_utils.to_categorical(y_train, len(category))
    y_test = np_utils.to_categorical(y_test, len(category))

    return X_train, X_test, y_train, y_test


class DataSet(object):
    def __init__(self, img_rows, img_cols):
        self.img_rows = img_rows
        self.img_cols = img_cols
        # create adaption data set if not exist
        if not os.path.exists(self.adaptation_dataset):
            print '\nStart create adaptation data set %sX%s' % (self.img_rows, self.img_cols)
            input_dataset_path = os.path.join(ROOT_DIR, 'dataset')
            reshape_images(input_dataset_path, self.adaptation_dataset, self.img_rows, self.img_cols)
            print '\nFinish create adaptation data set %sX%s' % (self.img_rows, self.img_cols)
        self.categories = os.listdir(self.adaptation_dataset)
        self.data = None

    def _load(self):
        if self.data is None:
            print '\nStart load'
            self.data = {}
            for idx, category in enumerate(self.categories):
                self.data[category] = {}
                category_path = os.path.join(self.adaptation_dataset, category)
                case_folder = os.listdir(category_path)
                for sub_folder in case_folder:
                    self.data[category][sub_folder] = {'frames': [], 'labels': None}
                    labels, frames = [], []
                    case_folder_path = os.path.join(self.adaptation_dataset, category, sub_folder)
                    images = os.listdir(case_folder_path)
                    for im in images:
                        im_path = os.path.join(case_folder_path, im)
                        frames.append(np.array(Image.open(im_path)).transpose(2, 0, 1))
                        labels.append(idx)
                    self.data[category][sub_folder]['frames'] = np.array(frames)
                    self.data[category][sub_folder]['labels'] = np.array(labels)
            print '\nFinish load'

            # Clear Memory
            gc.collect()

    @property
    def adaptation_dataset(self):
        return os.path.join(ROOT_DIR, 'dataset_' + str(self.img_rows) + 'X' + str(self.img_cols) + '_adaptation')

    def get_data_set_split_frames(self, train_ratio):
        self._load()
        all_frames, labels = [], []
        for c in self.categories:
            for k in self.data[c].keys():
                all_frames.extend(self.data[c][k]['frames'])
                labels.extend(self.data[c][k]['labels'])
        img_matrix, label = np.array(all_frames), np.array(labels)
        # random_state for psudo random
        data, label = shuffle(img_matrix, label, random_state=7)

        # the data set load, shuffled and split between train and validation sets
        X_train, X_test, y_train, y_test = train_test_split(data, label, test_size=1-train_ratio, random_state=7)

        return _normalized_data_set(X_train, X_test, y_train, y_test, self.categories)

    def get_data_set_split_cases(self, train_ratio):
        self._load()
        train_img_matrix, train_label, test_img_matrix, test_label = [], [], [], []
        for c in self.categories:
            case_folder = self.data[c].keys()
            for k in case_folder[0:int(len(case_folder) * train_ratio)]:
                train_img_matrix.extend(self.data[c][k]['frames'])
                train_label.extend(self.data[c][k]['labels'])
            for k in case_folder[int(len(case_folder)*train_ratio):]:
                test_img_matrix.extend(self.data[c][k]['frames'])
                test_label.extend(self.data[c][k]['labels'])
        train_img_matrix, train_label = shuffle(train_img_matrix, train_label, random_state=7)
        train_img_matrix, train_label = np.array(train_img_matrix), np.array(train_label)
        test_img_matrix, test_label = shuffle(test_img_matrix, test_label, random_state=7)
        test_img_matrix, test_label = np.array(test_img_matrix), np.array(test_label)
        return _normalized_data_set(train_img_matrix, test_img_matrix, train_label, test_label, self.categories)
