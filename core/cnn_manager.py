import os
from core.cnn import CNN
from manage import ROOT_DIR
from utils.singleton import singleton


@singleton
class CNNManager(object):
    def __init__(self):
        self.models = {}
        self.last_index = 0
        self.load_models()

    def load_models(self):
        models_names = set()
        file_list = os.listdir(os.path.join(ROOT_DIR, 'cnn_models'))
        for _file in file_list:
            model_name = _file.split('.')[0]
            models_names.add(model_name)
        for i in models_names:
            cnn = CNN()
            model_path = os.path.join(ROOT_DIR, 'cnn_models', i)
            cnn.load(model_path)
            self.models[i] = cnn

    def add_model(self, cnn, model_name):
        if model_name in self.models.keys():
            return False, 'Model name already exist'
        model_path = os.path.join(ROOT_DIR, 'cnn_models', model_name)
        cnn.save(model_path)
        self.models[model_name] = cnn
        return True, 'ok'

    def remove_model(self, model_name):
        cnn_model = self.models[model_name]
        del self.models[model_name]
        os.remove(cnn_model.model_path+'.h5')
        os.remove(cnn_model.model_path + '.json')

    def get_models(self):
        return {k: self.models[k].get_info() for k in self.models.keys()}
