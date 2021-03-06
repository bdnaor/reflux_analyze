import os
import json
import random
from manage import ROOT_DIR

RANDOM_CONFIGURATION_PATH = os.path.join(ROOT_DIR, 'random_conf.json')


def load_configurations():
    if not os.path.exists(RANDOM_CONFIGURATION_PATH):
        return []
    with open(RANDOM_CONFIGURATION_PATH, 'r') as _input:
        tmp = json.loads(_input.read())
        return tmp


def save_configurations(conf):
    with open(RANDOM_CONFIGURATION_PATH, 'w') as output:
        output.write(json.dumps(conf, sort_keys=True, indent=4, separators=(',', ': ')))


def get_random_conf():
    configurations = load_configurations()
    while True:
        params = dict()
        params['model_name'] = "random_%s" % len(configurations)
        params['split_cases'] = random.choice([True, False])
        params['dropout'] = random.choice([0.25, 0.5])
        params['activation_function'] = random.choice(['softmax', 'sigmoid'])
        img_size = random.choice([(50, 50), (75, 75), (100, 100)])  # (150, 150), (200, 200)
        params['img_rows'] = img_size[0]
        params['img_cols'] = img_size[1]
        params['sigma'] = random.uniform(0, 3)
        params['theta'] = random.uniform(0, 180)
        params['lammbd'] = random.uniform(0, 10)
        params['gamma'] = random.uniform(0, 1)
        params['psi'] = random.uniform(0, 90)
        params['nb_filters'] = random.choice([32, 64])
        params['kernel_size'] = random.choice([5, 6, 7, 8, 9, 10])
        params['pool_size'] = random.choice([2, 4, 6, 8])
        params['batch_size'] = random.choice([32, 64, 128])
        if params not in configurations:
            configurations.append(params)
            save_configurations(configurations)
            break
    return params
