import torch


class DataTransform:
    def __init__(self, is_train, features=['Timestamp', 'Open', 'High', 'Low', 'Close']):
        self.is_train = is_train
        self.keys = features
        self.use_volume = 'Volume' in self.keys
        print(self.keys)

    def __call__(self, window):
        data_list = []
        output = {}

        for key in self.keys:
            data = torch.tensor(window.get(key).tolist())
            if key == 'Volume':
                data /= 1e9
            output[key] = data[-1]
            output[f'{key}_old'] = data[-2]
            if key == 'Volume' and not self.use_volume:
                continue
            if key == 'Timestamp_orig':
                continue
            data_list.append(data[:-1].reshape(1, -1))
        features = torch.cat(data_list, 0)
        output['features'] = features
        # raise ValueError(output)
        return output

    def set_initial_seed(self, seed):
        self.rng.seed(seed)