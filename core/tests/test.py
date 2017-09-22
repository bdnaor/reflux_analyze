import random

from cv2.cv2 import CV_64F
# from skimage.filters import gabor_kernel
import keras.backend as K
import numpy as np
from keras.layers import Conv2D
from sklearn.model_selection import train_test_split
from sklearn.utils import shuffle
from keras.utils import np_utils
from keras.layers.convolutional import MaxPooling2D
from keras.layers.core import Dense, Dropout, Activation, Flatten
from keras.models import Sequential
import cv2
from theano import shared


images_size = 100
img_rows = 200
img_cols = 200
nb_channel = 3


def custom_gabor(shape, dtype=None):
    total_ker = []
    for i in xrange(shape[0]):
        kernels = []
        for j in xrange(shape[1]):
            # gk = gabor_kernel(frequency=0.2, bandwidth=0.1)
            tmp_filter = cv2.getGaborKernel(ksize=(shape[3], shape[2]), sigma=1, theta=1, lambd=0.5, gamma=0.3, psi=(3.14) * 0.5,
                               ktype=CV_64F)
            filter = []
            for row in tmp_filter:
                filter.append(np.delete(row, -1))
            kernels.append(filter)
                # gk.real
        total_ker.append(kernels)
    np_tot = shared(np.array(total_ker))
    return K.variable(np_tot, dtype=dtype)


def build_model():
    model = Sequential()
    # Layer 1
    model.add(Conv2D(32, (3, 3), kernel_initializer=custom_gabor,
                     input_shape=(nb_channel, img_rows, img_cols)))
    model.add(Activation('relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))

    # Layer 2
    model.add(Conv2D(32, (3, 3)))  # , kernel_initializer=custom_gabor
    model.add(Activation('relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))

    # Layer 3
    model.add(Conv2D(32, (3, 3)))  # , kernel_initializer=custom_gabor
    model.add(Activation('relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))

    model.add(Dropout(0.25))
    model.add(Flatten())
    model.add(Dense(64))
    model.add(Activation('relu'))
    model.add(Dropout(0.5))

    model.add(Dense(2))
    model.add(Activation('softmax'))

    model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
    return model


def make_dummy_data_set():
    img_matrix = []
    for i in xrange(images_size):
        img_matrix.append([random.random() for _ in xrange(img_rows*img_cols*nb_channel)])

    img_matrix = np.array(img_matrix)
    label = np.array([random.randint(0, 1) for _ in xrange(images_size)])

    data, label = shuffle(img_matrix, label, random_state=7)  # random_state=2
    X_train, X_test, y_train, y_test = train_test_split(data, label, test_size=0.2, random_state=7)

    # reshape the data
    X_train = X_train.reshape(X_train.shape[0], nb_channel, img_rows, img_cols)
    X_test = X_test.reshape(X_test.shape[0], nb_channel, img_rows, img_cols)

    X_train = X_train.astype('float32')
    X_test = X_test.astype('float32')

    # convert class vectore to binary class matrices
    y_train = np_utils.to_categorical(y_train, 2)
    y_test = np_utils.to_categorical(y_test, 2)

    return X_train, X_test, y_train, y_test


def train_model(model, X_train, X_test, y_train, y_test):
    model.fit(X_train,
              y_train,
              batch_size=32,
              epochs=5,
              verbose=1,
              validation_data=(X_test, y_test))


if __name__ == "__main__":
    model = build_model()
    # print model.layers[0].weights
    # print model.layers[0].get_weights[0].shape
    X_train, X_test, y_train, y_test = make_dummy_data_set()
    train_model(model, X_train, X_test, y_train, y_test)