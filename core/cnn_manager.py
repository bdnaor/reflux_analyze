import os
import re
from core.cnn import CNN
from manage import ROOT_DIR


class CNNManager(object):
    def __init__(self):
        self.models = []
        self.last_index = 0
        self.load_models()

    def load_models(self):
        models = set()
        file_list = os.listdir(os.path.join(ROOT_DIR, 'cnn_models'))
        for _file in file_list:
            model_name = _file.split('.')[0]
            models.add(model_name)
            num = int(re.findall(r'\d+', model_name)[-1])
            if num > self.last_index:
                self.last_index = num
        for i in models:
            cnn = CNN()
            model_path = os.path.join(ROOT_DIR, 'cnn_models', i)
            cnn.load(model_path)
            self.models.append(cnn)

    def add_model(self, cnn):
        self.models.append(cnn)
        self.last_index += 1
        model_name = 'cpu_cnn_model_%s' % self.last_index
        model_path = os.path.join(ROOT_DIR, 'cnn_models', model_name)
        cnn.save(model_path)

    def remove_model(self, cnn):
        if isinstance(cnn, int):
            cnn_model = self.models[cnn]
            del self.models[cnn]
        else:
            cnn_model = cnn
            self.models.remove(cnn)
        os.remove(cnn_model.model_path+'.h5')
        os.remove(cnn_model.model_path + '.json')
