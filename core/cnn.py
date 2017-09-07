import json
import os
import numpy as np

from random import randint
from keras.callbacks import ModelCheckpoint, LambdaCallback
from keras.layers import Conv2D
from sklearn.model_selection import train_test_split
from sklearn.utils import shuffle
from sklearn.metrics import confusion_matrix
from PIL import Image
from keras.utils import np_utils
from keras.layers.convolutional import MaxPooling2D
from keras.layers.core import Dense, Dropout, Activation, Flatten
from keras.models import Sequential, load_model

from manage import ROOT_DIR


class CNN(object):
    def __init__(self, params, reload=False):
        self.model_name = params['model_name']
        self.model_path = os.path.join(ROOT_DIR, 'cnn_models', self.model_name)
        if reload:
            self._load()
        else:
            # the original data set
            self.input_dataset_path = 'dataset'
            # the data set after resize
            self.adaptation_dtatset = os.path.join(ROOT_DIR, self.input_dataset_path +'_' +self.model_name +'_adaptation')

            self.img_rows = params.get('img_rows', 200)
            self.img_cols = params.get('img_cols', 200)
            self.nb_channel = params.get('nb_channel', 3)
            self.batch_size = params.get('batch_size', 32)
            self.epoch = params.get('epoch', 5)
            # number of convolution filter to use
            self.nb_filters = params.get('nb_filters', 32)
            # size of pooling area for max pooling
            pool_size = params.get('pool_size', 2)
            self.pool_size = (pool_size, pool_size)
            # convolution kernel size
            self.kernel_size = params.get('kernel_size', 3)
            self.dropout = params.get('dropout', 0.25)
            self.activation_function = params.get('activation_function','softmax')  # 'sigmoid'

        # History of the training
        self.hist = None
        self.category = []

        # the deep learning model
        self.model = None

        # the data set for train and validation
        self.X_train = None
        self.X_test = None
        self.y_train = None
        self.y_test = None

        self.load_data_set()
        self._build_model()

    def load_data_set(self):
        '''
        this method initialize the (X_train,y_train)(X_val, y_val) where X is the data and y is the label
        :return:
        '''

        # global nb_classes, X_train, X_test, y_train, y_test
        # get the categories according to the folder
        categories = os.listdir(self.adaptation_dtatset)
        # create matrix to store all images flatten
        img_matrix = []
        label = []
        for idx, category in enumerate(categories):
            self.category.append(category)
            category_path = os.path.join(self.adaptation_dtatset, category)
            sub_folders = os.listdir(category_path)
            for sub_folder in sub_folders:
                case_folder_path = os.path.join(self.adaptation_dtatset, category, sub_folder)
                images = os.listdir(case_folder_path)
                for im in images:
                    im_path = os.path.join(case_folder_path, im)
                    img_matrix.append(np.array(np.array(Image.open(im_path)).flatten()))
                    label.append(idx)

        img_matrix = np.array(img_matrix)
        label = np.array(label)

        # random_state for psudo random
        data, label = shuffle(img_matrix, label) #random_state=2
        X_train, X_test, y_train, y_test = train_test_split(data, label, test_size=0.2)

        # reshape the data
        X_train = X_train.reshape(X_train.shape[0], self.nb_channel, self.img_rows, self.img_cols)
        X_test = X_test.reshape(X_test.shape[0], self.nb_channel, self.img_rows, self.img_cols)

        self.X_train = X_train.astype('float32')
        self.X_test = X_test.astype('float32')

        # help for faster convert
        # self.X_train /= 255
        # self.X_test /= 255

        # convert class vectore to binary class matrices
        self.y_train = np_utils.to_categorical(y_train, len(self.category))
        self.y_test = np_utils.to_categorical(y_test, len(self.category))

        # print('X_train shape: ', self.X_train.shape)
        # print(self.X_train.shape[0], 'train example')
        # print(self.X_test.shape[0], 'validation example')

    def _build_model(self):
        # the data set load, shuffled and split between train and validation sets
        self.model = Sequential()

        # Layer 1
        self.model.add(Conv2D(filters=self.nb_filters,
                              kernel_size=self.kernel_size,
                              padding='valid',
                              input_shape=(self.nb_channel, self.img_rows, self.img_cols)))

        self.model.add(Activation('relu'))
        self.model.add(MaxPooling2D(pool_size=self.pool_size))

        # Layer 2
        self.model.add(Conv2D(filters=self.nb_filters, kernel_size=self.kernel_size, padding='valid'))
        self.model.add(Activation('relu'))
        self.model.add(MaxPooling2D(pool_size=self.pool_size))

        # Layer 3
        self.model.add(Conv2D(filters=self.nb_filters, kernel_size=self.kernel_size, padding='valid'))
        self.model.add(Activation('relu'))
        self.model.add(MaxPooling2D(pool_size=self.pool_size))

        self.model.add(Dropout(self.dropout))
        self.model.add(Flatten())
        self.model.add(Dense(64))
        self.model.add(Activation('relu'))
        self.model.add(Dropout(0.5))

        self.model.add(Dense(len(self.category)))
        self.model.add(Activation(self.activation_function))

        # rsm = optimizers.RMSprop(lr=0.001, rho=0.9, epsilon=1e-08, decay=0.0)

        # binary_accuracy, categorical_accuracy
        self.model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])

    def _print_confusion_matrix(self):
        y_pred = self.model.predict_classes(self.X_test)
        print confusion_matrix(np.argmax(self.y_test, axis=1), y_pred)

    def train_model(self):
        '''
            saves the model weights after each epoch if the validation loss decreased
        '''
        if self.X_train is None:
            self.load_data_set()

        checkpointer = ModelCheckpoint(filepath=self.model_path, verbose=1, save_best_only=True)
        confusion_matrix = LambdaCallback(on_epoch_end=lambda epoch, logs: self._print_confusion_matrix())
        hist = self.model.fit(self.X_train, self.y_train, batch_size=self.batch_size, epochs=self.epoch, verbose=1,
                              validation_data=(self.X_test, self.y_test), callbacks=[checkpointer, confusion_matrix])
        # hist = model.fit(X_train, y_train, batch_size=batch_size, nb_epoch=nb_epoch, verbose=1, validation_split=0.2)
        self.save(self.model_path)

    def save(self):
        del self.X_train
        del self.X_test
        del self.y_train
        del self.y_test
        self.model.save(self.model_path+'.h5')
        del self.model
        with open(self.model_path+'.json', 'wb') as output:
            output.write(json.dumps(self.__dict__))
            # pickle.dump(self, output, pickle.HIGHEST_PROTOCOL)

    def _load(self):
        with open(self.model_path+'.json', 'rb') as input:
            # tmp = pickle.load(input)
            tmp = json.loads(input.read())
        self.__dict__ = tmp
        self.model = load_model(self.model_path + '.h5')

    def predict(self, frame):
        frame = np.array(np.array(Image.open(frame)).flatten())
        frame = frame.reshape(1, self.nb_channel, self.img_rows, self.img_cols)
        pred = self.model.predict(frame, batch_size=1)
        return self.category[0] if pred[0][0] > pred[0][1] else self.category[1]

    def get_random_frame(self):
        categories = os.listdir(self.adaptation_dtatset)
        category = randint(0, self.nb_classes-1)
        category_path = os.path.join(self.adaptation_dtatset, categories[category])
        cases = os.listdir(category_path)
        case_index = randint(0, len(cases)-1)
        frames = os.listdir(os.path.join(category_path, cases[case_index]))
        frame_index = randint(0, len(frames)-1)
        random_frame = os.path.join(category_path, cases[case_index], frames[frame_index])
        return random_frame, category

    def get_info(self):
        return {
            'img_rows': self.img_rows,
            'img_cols': self.img_cols,
            'epoch': self.epoch,
            'pool_size': self.pool_size[0],
            'kernel_size': self.kernel_size,
        }
