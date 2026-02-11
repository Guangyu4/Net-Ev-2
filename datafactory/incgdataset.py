import os, json
import torch
import ast
import numpy as np
import pandas as pd
from torch.utils.data import Dataset
from sklearn.preprocessing import MinMaxScaler

class INCGDataset(Dataset):
    def __init__(
            self,
            name='Agriculture',
            data_root='./Data/MMD',
            window=24,
            proportion=0.99,
            seed=123,
            period='train',
            max_length=32,
            args=None,
    ):
        super(INCGDataset, self).__init__()
        assert period in ['train', 'test'], 'period must be train or test.'
        
        self.data_root = data_root
        self.year = args.year
        self.period = period
        self.train_data_path = os.path.join(data_root, args.year, 'train.npz')
        self.test_data_path = os.path.join(data_root, args.year, 'test.npz')
        
        with np.load(self.train_data_path) as train_file:
            self.train_length = train_file['data'].shape[0]
        with np.load(self.test_data_path) as test_file:
            self.test_length = test_file['data'].shape[0]
        
        self.train_mmap = None
        self.test_mmap = None
        
        if period == 'train':
            with open(f'{data_root}/{args.year}/train.json', 'r') as f:
                print('Loading train metadata.......................')
                self.Data = json.load(f)
        else:
            with open(f'{data_root}/{args.year}/test.json', 'r') as f:
                print('Loading test metadata.......................')
                self.Data = json.load(f)
        
        self._length = len(self.Data)
    
    def _get_memory_mapped_data(self):
        if self.period == 'train' and self.train_mmap is None:
            train_data = np.load(self.train_data_path)
            self.train_mmap = train_data['data']
        if self.period == 'test' and self.test_mmap is None:
            test_data = np.load(self.test_data_path)
            self.test_mmap = test_data['data']
    
    def __getitem__(self, ind):
        example = dict(self.Data[ind])
        
        self._get_memory_mapped_data()
        
        if self.period == 'train':
            x = self.train_mmap[ind, :, :]
        else:
            x = self.test_mmap[ind, :, :]
        
        if np.isnan(x).any():
            x = np.nan_to_num(x, nan=0.0)
        
        text = str(example['text'])
        nodeid = example['nodelist']
        timeid = example['event_count']
        return text, x, nodeid, timeid

    def __len__(self):
        return self._length