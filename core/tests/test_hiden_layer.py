from unittest import TestCase
from core.cnn import CNN
import numpy as np

from PIL import Image


class TestVisualHidenLayer(TestCase):
    def test(self):
        im_path = "/home/naor/PycharmProjects/reflux_analyze/dataset_75X75_adaptation/negative/006/augmentation5"
        im = Image.open(im_path)
        # pixels = list(im.getdata())
        # width, height = im.size
        # pixels = [pixels[i * width:(i + 1) * width] for i in xrange(height)]
        x = np.array(im).flatten()
        print x
        outimg = Image.fromarray(x, "RGB")
        outimg.save("/tmp/ycc.png")

        # cnn = CNN({'model_name': 'model9'}, True)
        # f, r = cnn.get_random_frame()
        # cnn.get_activations(2, f)








'''
Model Layers
00 = {Conv2D} <keras.layers.convolutional.Conv2D object at 0x7fae4b185550>
01 = {Activation} <keras.layers.core.Activation object at 0x7fae4a541b10>
02 = {MaxPooling2D} <keras.layers.pooling.MaxPooling2D object at 0x7fae4a541b50>

03 = {Conv2D} <keras.layers.convolutional.Conv2D object at 0x7fae4a4da710>
04 = {Activation} <keras.layers.core.Activation object at 0x7fae4a509f90>
05 = {MaxPooling2D} <keras.layers.pooling.MaxPooling2D object at 0x7fae4a477750>

06 = {Dropout} <keras.layers.core.Dropout object at 0x7fae4a477390>
07 = {Flatten} <keras.layers.core.Flatten object at 0x7fae4a45acd0>
08 = {Dense} <keras.layers.core.Dense object at 0x7fae4a1cf950>
09 = {Activation} <keras.layers.core.Activation object at 0x7fae4a1da890>
10 = {Dropout} <keras.layers.core.Dropout object at 0x7fae49fbf710>
11 = {Dense} <keras.layers.core.Dense object at 0x7fae48d2e550>
12 = {Activation} <keras.layers.core.Activation object at 0x7fae48d3b510>




img = Image.fromarray(activations[0][15], 'RGB')
img.save('/tmp/my.png')
'''