import os
import torch
import time
import random
import pandas as pd
from tqdm import tqdm
from pathlib import Path
from utils import io_tools
from datetime import datetime
from utils.io_tools import load_config_from_yaml

    
class CMambaDataset(torch.utils.data.Dataset):

    def __init__(
        self,
        data,
        split,
        window_size,
        transform,
    ):

        self.data = data
        self.transform = transform
        self.window_size = window_size
            
        print('{} data points loaded as {} split.'.format(len(self), split))

    def __len__(self):
        return max(0, len(self.data) - self.window_size - 1)

    def __getitem__(self, i: int):
        sample = self.data.iloc[i: i + self.window_size + 1]
        sample = self.transform(sample)
        return sample
    
class DataConverter:
    def __init__(self, config) -> None:
        self.config = config
        self.root = config.get('root')
        self.jumps = config.get('jumps') * 60
        self.end_date = config.get('end_date')
        self.data_path = config.get('data_path')
        self.start_date = config.get('start_date')
        self.folder_name = f'{self.start_date}_{self.end_date}_{self.jumps // 60}'
        self.file_path = f'{self.root}/{self.folder_name}'

    
    def process_data(self):
        data, start, stop = self.load_data()
        new_df = {}
        for key in data.keys():
            if key == 'Timestamp_orig':
                continue
            if key == 'Date':
                continue
            
            new_df[key] = []
        for i in tqdm(range(start, stop - 3600, self.jumps)):
            merged_data = self.merge_data(data, i, self.jumps)

            if merged_data is None:
                print(f'No data found for timestamp {i}')
                continue

            for key in new_df.keys():
                if key == 'Timestamp':
                    new_df.get(key).append(i)
                else:
                    new_df.get(key).append(float(merged_data.get(key)))

        df = pd.DataFrame(new_df)

        return df

    def get_data(self):
        tmp = '----'
        data_path = f'{self.file_path}/{tmp}.csv'
        yaml_path = f'{self.file_path}/config.pkl'
        if os.path.isfile(data_path.replace(tmp, 'train')):
            train = pd.read_csv(data_path.replace(tmp, 'train'), index_col=0)
            val = pd.read_csv(data_path.replace(tmp, 'val'), index_col=0)
            test = pd.read_csv(data_path.replace(tmp, 'test'), index_col=0)
            return train, val, test
        
        df = self.process_data()

        train, val, test = self.split(df)

        if not os.path.isdir(self.file_path):
            os.mkdir(self.file_path)

        train.to_csv(data_path.replace('----', 'train'))
        val.to_csv(data_path.replace('----', 'val'))
        test.to_csv(data_path.replace('----', 'test'))
        io_tools.save_yaml(self.config, yaml_path)
        return train, val, test
    

    def split(self, data):
        if self.config.get('train_ratio') is not None:
            total = len(data)
            n_train = int(self.train_ratio * total)
            n_test = int(self.test_ratio * total)
            total = list(range(total))
            random.shuffle(total)
        
            train = sorted(total[: n_train])
            val = sorted(total[n_train: -n_test])
            test = sorted(total[-n_test: ])
            train = data.iloc[train]
            val = data.iloc[val]
            test = data.iloc[test]
        else:
            tmp_dict = {}
            for key in ['train', 'val', 'test']:
                start, stop = self.config.get(f'{key}_interval')
                start = self.generate_timestamp(f'{start}: 00-00')
                stop = self.generate_timestamp(f'{stop}: 00-00')
                tmp = data[data['Timestamp'] >= start].reset_index(drop=True)
                tmp = tmp[tmp['Timestamp'] < stop].reset_index(drop=True)
                tmp_dict[key] = tmp
            train = tmp_dict.get('train')
            val = tmp_dict.get('val')
            test = tmp_dict.get('test')
        return train, val, test


    def load_data(self):
        df = pd.read_csv(self.data_path)
        if 'Timestamp' not in df.keys():
            if 'Date' not in df.keys():
                print("no date in dataset")
                return

            dates = df.get('Date').to_list()
            df['Timestamp'] = [self.generate_timestamp(x, d_format="%Y-%m-%d") for x in dates]
        if self.start_date is None:
            self.start_date = self.convert_timestamp(min(list(df.get('Timestamp')))).strftime('%Y-%d-%m')
            # raise ValueError(self.start_date)
        if self.end_date is None:
            self.end_date = self.convert_timestamp(max(list(df.get('Timestamp'))) + 24 * 60 * 60).strftime('%Y-%d-%m')
        start = self.generate_timestamp(f'{self.start_date}: 00-00')
        stop = self.generate_timestamp(f'{self.end_date}: 00-00')
        df = df[df['Timestamp'] >= start].reset_index(drop=True)
        df = df[df['Timestamp'] < stop].reset_index(drop=True)
        final_day = self.generate_timestamp(f'{self.end_date}: 00-00')
        return df, start, final_day
    
    @staticmethod
    def merge_data(data, start, jump):
        tmp = data[data['Timestamp'] >= start].reset_index(drop=True)
        tmp = tmp[tmp['Timestamp'] < start + jump].reset_index(drop=True)
        if len(tmp) == 0:
            return None #, None, None, None, None
        if len(tmp) == 1:
            return tmp.iloc[0]
            # _, high, low, open, close, volume = DataConverter.get_row_values(tmp.iloc[0])
            # return high, low, open, close, volume

        print("\033[91mMore than one row\033[0m")

        _, high, low, open, close, _ = DataConverter.get_row_values(tmp.iloc[0])
        count = 1
        vol = 0
        for row in tmp.iterrows():
            _, h, l, _, close, v = DataConverter.get_row_values(row[1])
            high = max(high, h)
            low = min(low, l)
            vol += v
            count += 1
        
        return high, low, open, close, vol
    
    @staticmethod
    def get_row_values(row):
        ts = int(row.get('Timestamp'))
        high = float(row.get('High'))
        low = float(row.get('Low'))
        open = float(row.get('Open'))
        close = float(row.get('Close'))
        vol = float(row.get('Volume'))
        return ts, high, low, open, close, vol

    @staticmethod
    def generate_timestamp(date, d_format="%Y-%d-%m: %H-%M"):
        """
            Year-day-month: hour-minute (all zero padded)
        """
        return int(time.mktime(datetime.strptime(date, d_format).timetuple()))

    @staticmethod
    def convert_timestamp(timestamp):
        return datetime.fromtimestamp(timestamp)