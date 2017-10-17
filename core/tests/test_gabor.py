from unittest import TestCase
from core.cnn import CNN


class TestLoadSaveLoadModel(TestCase):
    def test(self):
        cnn = CNN({'model_name': 'item 1'}, True)
        cnn.sigma = 0.1
        cnn.gamma = 0.1     # 0-1
        cnn.theta = 0.3
        cnn.lambd = 0.4
        cnn.psi = 0.5
        cg = cnn.get_custom_gabor()
        ker = cg((3, 3, 3, 3))
        print ker.container