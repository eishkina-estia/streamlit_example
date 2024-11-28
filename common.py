import os
import yaml

# project root
ROOT_DIR = os.path.dirname(os.path.abspath(__file__))
# full config path
CONFIG_PATH = os.path.join(ROOT_DIR, 'config.yml')

CONFIG_LIST_SEP = ";"

def get_full_path(rel_path):
    return os.path.normpath(os.path.join(ROOT_DIR, rel_path))

def get_dict_from_config_value(s):
    values = s.split(CONFIG_LIST_SEP)
    keys = range(len(values))
    return dict(zip(keys, values))

with open(CONFIG_PATH, "r") as f:

    # yaml to python dict
    CONFIG = yaml.load(f, Loader=yaml.SafeLoader)

    for key, value in CONFIG['paths'].items():
        CONFIG['paths'][key] = get_full_path(value)
    # print(f"data_path: {CONFIG['paths']['data_path']}")

    CONFIG['ml']['target_labels'] = get_dict_from_config_value(CONFIG['ml']['target_labels'])
    # print(f"target_labels: {CONFIG['ml']['target_labels']}")