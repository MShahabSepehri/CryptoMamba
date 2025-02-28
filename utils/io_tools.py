import yaml
import pickle
import pathlib
import importlib
import pandas as pd

def get_obj_from_str(string, reload=False):
    module, cls = string.rsplit(".", 1)
    if reload:
        module_imp = importlib.import_module(module)
        importlib.reload(module_imp)
    return getattr(importlib.import_module(module, package=None), cls)


def instantiate_from_config(config):
    if not "target" in config:
        if config == '__is_first_stage__':
            return None
        elif config == "__is_unconditional__":
            return None
        raise KeyError("Expected key `target` to instantiate.")
    return get_obj_from_str(config["target"])(**config.get("params", dict()))


def get_root(file, num_returns=1):
    tmp = pathlib.Path(file)
    for _ in range(num_returns):
        tmp = tmp.parent.resolve()
    return tmp

def load_config_from_yaml(path):
    config_file = pathlib.Path(path)
    if config_file.exists():
        with config_file.open('r') as f:
            d = yaml.safe_load(f)
        return d
    else:
        raise ValueError(f'Config file ({path}) does not exist.')
    

def save_yaml(data, path):
    with open(path, 'w') as file:
        yaml.dump(data, file, default_flow_style=False)


def save_pickle(data, path: str) -> None:
    with open(path, 'wb') as f:
        pickle.dump(data, f)


def load_pickle(path: str):
    with open(path, 'rb') as f:
        data = pickle.load(f)
    return data

def get_train_csv_columns(data_config):
    file_path = f"{data_config.get('root')}/{data_config.get('start_date')}_{data_config.get('end_date')}_{data_config.get('jumps')}/train.csv"
    data = pd.read_csv(file_path, nrows=0)
    return [col for col in data.columns if not col.startswith('Unnamed')]
