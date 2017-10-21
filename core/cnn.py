import os
import gc
import traceback
import json
import cv2
import io
from cv2.cv2 import CV_64F
from random import randint
from datetime import datetime
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.utils import shuffle
from sklearn.metrics import confusion_matrix
from keras.callbacks import LambdaCallback
from keras.layers import Conv2D
from keras.utils import np_utils
from keras.layers.convolutional import MaxPooling2D
from keras.layers.core import Dense, Dropout, Activation, Flatten
from keras.models import Sequential, load_model
import keras.backend as K
from theano import shared
from PIL import Image
import StringIO
import base64
from scipy.misc import toimage

from core.data_set_manager import DataSetManager
from manage import ROOT_DIR


FORMAT = '%Y-%m-%d %H:%M:%S'


data_set_manager = DataSetManager()


def calculate_score(_confusion_matrix):
    try:
        tn, fp, fn, tp = _confusion_matrix
        _recall = float(tp) / (tp + fn)
        _precision = float(tp) / (tp+fp)
        return float(2) / ((float(1) / _recall) + (float(1) / _precision))
    except ZeroDivisionError:
        return 0
    except Exception as e:
        print e.message
        print traceback.format_exc()
        print 'Error: can not calculate score:\ntn %s\nfp %s\nfn %s\n tp %s\n' % (tn, fp, fn, tp)
        return 0


class CNN(object):
    def __init__(self, params, _reload=False):
        self.model_name = params['model_name']
        self.model_path = os.path.join(ROOT_DIR, 'cnn_models', self.model_name)
        self.input_dataset_path = os.path.join(ROOT_DIR, 'dataset')  # the original data set
        self.model = None  # the deep learning model
        if _reload:
            self._load()
        else:
            self.train_ratio = float(params.get('train_ratio', 0.5))
            self.split_cases = params.get('split_cases', True)
            if type(self.split_cases) is not bool:
                if self.split_cases.lower() == "false" or self.split_cases == False:
                    self.split_cases = False
                else:
                    self.split_cases = True
            self.img_rows = int(params.get('img_rows', 200))
            self.img_cols = int(params.get('img_cols', 200))
            self.nb_channel = int(params.get('nb_channel', 3))
            self.batch_size = int(params.get('batch_size', 32))
            # self.epoch = int(params.get('epoch', 5))
            self.nb_filters = int(params.get('nb_filters', 32))  # number of convolution filter to use
            self.dropout = float(params.get('dropout', 0.25))
            self.activation_function = params.get('activation_function', 'softmax')  # 'sigmoid'
            # Gabor params
            self.sigma = float(params.get('sigma', 1))
            self.theta = float(params.get('theta', 1))
            self.lambd = float(params.get('lambd', 0.5))
            self.gamma = float(params.get('gamma', 0.3))
            self.psi = float(params.get('psi', 1.57))

            self.con_mat_val = []
            self.con_mat_train = []
            self.hist = None  # History of the training
            self.times_start_test = []
            self.times_start_train = []
            self.times_finish = []
            self.category = ["negative", "positive"]
            self.total_train_epoch = 0
            self.done_train_epoch = 0
            self.index_best = 0
            # the data set for train and validation
            self.X_train = None
            self.X_test = None
            self.y_train = None
            self.y_test = None

            # size of pooling area for max pooling
            self.pool_size = params.get('pool_size', (2, 2))
            if isinstance(self.pool_size, unicode):
                self.pool_size = int(self.pool_size)
            if isinstance(self.pool_size, int):
                self.pool_size = (self.pool_size, self.pool_size)

            # convolution kernel size
            self.kernel_size = params.get('kernel_size', (3, 3))
            if isinstance(self.kernel_size, unicode):
                self.kernel_size = int(self.kernel_size)
            if isinstance(self.kernel_size, int):
                self.kernel_size = (self.kernel_size, self.kernel_size)
            self.with_gabor = params.get('with_gabor', True)
            if type(self.with_gabor) is not bool:
                if self.with_gabor.lower() == "false" or self.with_gabor == False:
                    self.with_gabor = False
                else:
                    self.with_gabor = True

            self._build_model()
        # self.load_datasets()

    def get_custom_gabor(self):
        def custom_gabor(shape, dtype=None):
            total_ker = []
            for t in np.arange(0, np.pi, np.pi /shape[3]):
                kernels = []
                for z in np.arange(0, np.pi, np.pi /shape[2]):
                    tmp_filter = cv2.getGaborKernel(ksize=(shape[0], shape[1]),
                                                    sigma=self.sigma,
                                                    theta=self.theta + t,
                                                    lambd=self.lambd,
                                                    gamma=self.gamma + z,
                                                    psi=self.psi,
                                                    ktype=CV_64F)
                    _filter = []
                    if shape[0]%2 == 0:
                        for i in xrange(len(tmp_filter)-1):
                            _filter.append(tmp_filter[i][0: shape[1]])
                    else:
                        _filter = tmp_filter
                    kernels.append(_filter)
                total_ker.append(kernels)
            np_tot = shared(np.array(total_ker).reshape(shape))
            return K.variable(np_tot, dtype=dtype)
        return custom_gabor

    def _build_model(self):

        self.model = Sequential()

        # Layer 1
        self.model.add(Conv2D(filters=self.nb_filters,
                              kernel_size=self.kernel_size,
                              padding='same',
                              kernel_initializer=self.get_custom_gabor() if self.with_gabor else 'random_normal',
                              input_shape=(self.nb_channel, self.img_rows, self.img_cols)))
        self.model.add(Activation('relu'))
        self.model.add(MaxPooling2D(pool_size=self.pool_size))

        # Layer 2
        self.model.add(Conv2D(filters=self.nb_filters*2,
                              kernel_size=self.kernel_size,
                              kernel_initializer=self.get_custom_gabor() if self.with_gabor else 'random_normal',
                              padding='same'))
        self.model.add(Activation('relu'))
        self.model.add(MaxPooling2D(pool_size=self.pool_size))

        # # Layer 3
        self.model.add(Dropout(self.dropout))
        self.model.add(Flatten())

        # Layer 4
        self.model.add(Dense(64))
        self.model.add(Activation('relu'))
        self.model.add(Dropout(self.dropout))

        # Layer 5
        self.model.add(Dense(len(self.category)))
        self.model.add(Activation(self.activation_function))

        self.model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])

    def _get_avg_score_list(self):
        avg_scores = [];
        for i in xrange(len(self.con_mat_val)):
            train_score = calculate_score(self.con_mat_train[i])
            test_score = calculate_score(self.con_mat_val[i])
            avg_scores.append((train_score+test_score)/2)
        return avg_scores

    def get_activations(self, frame_path):
        frame = np.array(Image.open(frame_path)).transpose(2, 0, 1)
        y = [1]
        y.extend(frame.shape)
        frame = frame.reshape(tuple(y))
        inp = self.model.input  # input placeholder
        outputs = [_layer.output for _layer in self.model.layers]  # all layer outputs
        functor = K.function([inp] + [K.learning_phase()], outputs)  # evaluation function
        # get_activations = K.function([self.model.layers[0].input, K.learning_phase()], self.model.layers[layer].output)
        # activations = get_activations([frame, 1])
        layer_outs = functor([frame, 0.])
        imgs = {0: [], 3: []}
        for lay in imgs.keys():
            activations = layer_outs[lay]
            for i in range(len(activations[0])):
                # img = Image.fromarray(activations[0][i])  # 'RGB'
                img = toimage(activations[0][i])
                in_mem_file = io.BytesIO()
                # if img.mode != 'RGB':
                #     img = img.convert('RGB')
                img.save(in_mem_file, format="PNG")
                # reset file pointer to start
                in_mem_file.seek(0)
                img_bytes = in_mem_file.read()
                imgs[lay].append("data:image/png;base64,%s" % (base64.b64encode(img_bytes),))
        return imgs

    def _save_only_best(self, epoch=None, logs=None):
        avg_scores = self._get_avg_score_list()
        last_biggest = True
        for i in xrange(len(avg_scores)-1):
            if avg_scores[i] > avg_scores[-1]:
                last_biggest = False
                break
        if last_biggest:
            self.index_best = len(avg_scores)-1
            print '\nfind best model and save it. avg_scores = %s' % (avg_scores[-1],)
            self.save()
        else:
            print '\nsave json only'
            self.save(only_json=True)

    def train_model(self, n_epoch=None):
        if self.split_cases:
            X_train, X_test, y_train, y_test = data_set_manager.get_data_set_split_cases(self.img_rows, self.img_cols, self.train_ratio)
        else:
            X_train, X_test, y_train, y_test = data_set_manager.get_data_set_split_frames(self.img_rows, self.img_cols, self.train_ratio)

        def _calculate_confusion_matrix(epoch=None, logs=None):
            try:
                # For test set
                if len(self.times_start_test) < 3:
                    self.times_start_test.append(datetime.now().strftime(FORMAT))
                y_pred = self.model.predict_classes(X_test)
                tn, fp, fn, tp = confusion_matrix(np.argmax(y_test, axis=1), y_pred).ravel()
                print "\nval: tn:%s, fp:%s, fn:%s, tp:%s" % (tn, fp, fn, tp)
                self.con_mat_val.append([tn, fp, fn, tp])

                # For train set
                if len(self.times_start_train) < 3:
                    self.times_start_train.append(datetime.now().strftime(FORMAT))
                y_pred = self.model.predict_classes(X_train)
                tn, fp, fn, tp = confusion_matrix(np.argmax(y_train, axis=1), y_pred).ravel()
                print "\ntrain: tn:%s, fp:%s, fn:%s, tp:%s" % (tn, fp, fn, tp)
                self.con_mat_train.append([tn, fp, fn, tp])

                if len(self.times_finish) < 3:
                    self.times_finish.append(datetime.now().strftime(FORMAT))
                self.done_train_epoch += 1
            except Exception as e:
                print traceback.format_exc()

        # Initialize params for progress bar
        self.done_train_epoch = 0
        self.total_train_epoch = n_epoch

        # Evaluate the created model at the first time only
        if not self.con_mat_val:
            _calculate_confusion_matrix()
            self._save_only_best()

        # Start train the model
        conf_matrix = LambdaCallback(on_epoch_end=lambda epoch, logs: _calculate_confusion_matrix(epoch, logs))
        save_only_best = LambdaCallback(on_epoch_end=lambda epoch, logs: self._save_only_best(epoch, logs))

        self.hist = self.model.fit(X_train,
                                   y_train,
                                   batch_size=self.batch_size,
                                   epochs=n_epoch,
                                   verbose=1,
                                   validation_data=(X_test, y_test),
                                   callbacks=[conf_matrix, save_only_best])

    def save(self, only_json=False):
        if not only_json:
            self.model.save_weights(self.model_path + '.h5(weights)')
        save_dict = self.get_info()
        with open(self.model_path+'.json', 'wb') as output:
            output.write(json.dumps(save_dict, sort_keys=True, indent=4, separators=(',', ': ')))

    def _find_index_best(self):
        avg_scores = self._get_avg_score_list()
        return avg_scores.index(min(avg_scores))

    def _load(self):
        with open(self.model_path+'.json', 'rb') as _input:
            tmp = json.loads(_input.read())
        self.__dict__.update(tmp)
        self.total_train_epoch = 0
        self.done_train_epoch = 0
        if not hasattr(self, 'times_start_test'):
            self.times_start_test = []
        if not hasattr(self, 'times_start_train'):
            self.times_start_train = []
        if not hasattr(self, 'times_finish'):
            self.times_finish = []
        if not hasattr(self, 'train_ratio'):
            self.train_ratio = 0.5
        if not hasattr(self, 'sigma'):
            self.sigma = 1
        if not hasattr(self, 'theta'):
            self.theta = 1
        if not hasattr(self, 'lambd'):
            self.lambd = 0.5
        if not hasattr(self, 'gamma'):
            self.gamma = 0.3
        if not hasattr(self, 'psi'):
            self.psi = 1.57
        if not hasattr(self, 'index_best'):
            self.index_best = self._find_index_best()

        if hasattr(self, 'with_gabor') and self.with_gabor:
            self._build_model()
            if os.path.exists(self.model_path + '.h5(weights)'):
                self.model.load_weights(self.model_path + '.h5(weights)')
        elif os.path.exists(self.model_path + '.h5(best)'):
            self.model = load_model(self.model_path + '.h5(best)')
        elif os.path.exists(self.model_path + '.h5'):
            self.model = load_model(self.model_path + '.h5')
        else:
            self._build_model()

    def predict(self, frame):
        img = Image.open(frame)
        if img.size != (self.img_cols, self.img_rows):
            raise Exception('Image Size Don\'t Matching.')
        frame = np.array(np.array(img).flatten())
        frame = frame.reshape(1, self.nb_channel, self.img_rows, self.img_cols)
        pred = self.model.predict(frame, batch_size=1)
        return self.category[0] if pred[0][0] > pred[0][1] else self.category[1]

    def get_info(self):
        return {
            "category": self.category,
            "nb_channel": self.nb_channel,
            "activation_function": self.activation_function,
            "dropout": self.dropout,
            "nb_filters": self.nb_filters,
            "pool_size": self.pool_size,
            "hist": self.hist,
            "train_ratio": self.train_ratio,
            "split_cases": self.split_cases,
            "img_rows": self.img_rows,
            "img_cols": self.img_cols,
            "batch_size": self.batch_size,
            "con_mat_train": self.con_mat_train,
            "con_mat_val": self.con_mat_val,
            "model_name": self.model_name,
            "kernel_size": self.kernel_size,
            "total_train_epoch": self.total_train_epoch,
            "done_train_epoch": self.done_train_epoch,
            "with_gabor": self.with_gabor,
            "sigma": self.sigma,
            "theta": self.theta,
            "lambd": self.lambd,
            "gamma": self.gamma,
            "psi": self.psi,
            "times_start_test": self.times_start_test,
            "times_start_train": self.times_start_train,
            "times_finish": self.times_finish,
            "index_best": self.index_best
        }

    def get_random_frame(self):
        # if not hasattr(self, 'adaptation_dataset'):
        #     self.load_datasets()
        categories = data_set_manager.get_categories(self.img_rows, self.img_cols)
        adaptation_dataset = data_set_manager.get_adaptation_dataset(self.img_rows, self.img_cols)
        category = randint(0, len(categories)-1)
        category_path = os.path.join(adaptation_dataset, categories[category])
        cases = os.listdir(category_path)
        case_index = randint(0, len(cases)-1)
        frames = os.listdir(os.path.join(category_path, cases[case_index]))
        frame_index = randint(0, len(frames)-1)
        random_frame = os.path.join(category_path, cases[case_index], frames[frame_index])
        return random_frame, categories[category]

    def get_random_prediction(self):
        random_frame, real = self.get_random_frame()
        prediction = self.predict(random_frame)
        img = open(random_frame, "rb").read()
        img = base64.b64encode(img)
        L_Out = self.get_activations(random_frame)
        # img = Image.open(random_frame)
        return {'img': img, 'prediction': prediction, 'real': real, 'L_Out': L_Out}

    def create_model_svg(self):
        # from IPython.display import SVG
        from keras.utils.vis_utils import model_to_dot

        # SVG(model_to_dot(model).create(prog='dot', format='svg'))
        svg_res = model_to_dot(self.model).create(prog='dot', format='svg')
        with open(self.model_path+'.svg','w') as _f:
            _f.write(svg_res)
