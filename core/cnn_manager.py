import os
import shutil

from core.cnn import CNN
from manage import ROOT_DIR
from utils.singleton import singleton


@singleton
class CNNManager(object):
    def __init__(self):
        self.models = {}
        self.load_models()

    def load_models(self):
        models_names = set()
        file_list = os.listdir(os.path.join(ROOT_DIR, 'cnn_models'))
        for _file in file_list:
            model_name = _file.split('.')[0]
            models_names.add(model_name)
        for i in models_names:
            print 'load model %s' % i
            cnn = CNN({'model_name': i}, True)
            self.models[i] = cnn

    def add_model(self, cnn):
        if cnn.model_name in self.models.keys():
            return False, 'Model name already exist'
        cnn.save()
        self.models[cnn.model_name] = cnn
        return True, 'ok'

    def remove_model(self, model_name):
        cnn_model = self.models[model_name]
        del self.models[model_name]
        os.remove(cnn_model.model_path+'.h5')
        os.remove(cnn_model.model_path + '.json')
        shutil.rmtree(cnn_model.adaptation_dtatset)

    def get_models(self):
        return {k: self.models[k].get_info() for k in self.models.keys()}

    def get_random_frame(self, model_name):
        cnn_model = self.models[model_name]
        return cnn_model.get_random_prediction()