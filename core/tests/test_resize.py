from unittest import TestCase

from core.cnn_manager import CNNManager
from PIL import Image
from PIL.Image import LANCZOS

class TestPredictFrame(TestCase):
    def test_predict(self):
        image_path_in = "/home/naor/PycharmProjects/reflux_analyze/dataset/negative/004/augmentation5.png"
        image_path_out = "/tmp/tt"
        im = Image.open(image_path_in)
        img = im.resize((75, 75), resample=LANCZOS)
        # img = img.convert('L')
        # gray = img.convert('L')d
        # gray.save(output_dtatset+'\\'+f,'JPEG')
        img.save(image_path_out, 'JPEG')