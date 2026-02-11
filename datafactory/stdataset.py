import os
import numpy as np
import h5py
from torch.utils.data import Dataset
from tqdm import tqdm

class STDataset(Dataset):
    def __init__(
            self,
            name='SD',
            data_root='./Data',
            window=96,
            proportion=0.99,
            seed=123,
            period='train',
            max_length=32,
            args=None,
            preload=True,
    ):
        super(STDataset, self).__init__()
        assert period in ['train', 'test'], 'period must be train or test.'
        
        self.data_root = data_root
        self.year = args.year
        self.period = period
        self.preload = preload
        
        h5_path = os.path.join(data_root, args.year, f'{period}.h5')
        print(f'Loading {period} data from {h5_path}...')
        
        self.h5_file = h5py.File(h5_path, 'r')
        self._length = self.h5_file.attrs['n_samples']
        self.mean = self.h5_file.attrs['mean']
        self.std = self.h5_file.attrs['std']
        
        if preload:
            print(f'Preloading {self._length} samples to memory...')
            self.data = np.nan_to_num(self.h5_file['data'][:], nan=0.0)
            self.nodeid = self.h5_file['nodeid'][:]
            self.nodelist = self.h5_file['nodelist'][:]
            self.event_type = self.h5_file['event_type'][:]
            self.event_count = self.h5_file['event_count'][:]
            self.weather_type = self.h5_file['weather_type'][:]
            self.weather_count = self.h5_file['weather_count'][:]
            self.text_list = []
            for i in tqdm(range(self._length), desc='Building text'):
                event_type = self.event_type[i]
                weather_type = self.weather_type[i]
                event_desc = self.h5_file['event_description'][i]
                weather_desc = self.h5_file['weather_description'][i]
                text_parts = []
                if event_type != -1:
                    text_parts.append(f"Event: {event_desc}")
                if weather_type != 0:
                    text_parts.append(f"Weather: {weather_desc}")
                text = ". ".join(text_parts) if text_parts else "Normal traffic conditions."
                self.text_list.append(text)
            self.h5_file.close()
            self.h5_file = None
        else:
            self.data = self.h5_file['data']
            self.nodeid = self.h5_file['nodeid']
            self.nodelist = self.h5_file['nodelist']
            self.event_type = self.h5_file['event_type']
            self.event_count = self.h5_file['event_count']
            self.event_description = self.h5_file['event_description']
            self.weather_type = self.h5_file['weather_type']
            self.weather_count = self.h5_file['weather_count']
            self.weather_description = self.h5_file['weather_description']
    
    def __getitem__(self, ind):
        x = self.data[ind]
        if np.isnan(x).any():
            x = np.nan_to_num(x, nan=0.0)
        nodelist = self.nodelist[ind]
        
        if self.preload:
            text = self.text_list[ind]
        else:
            # lazy loading
            event_type = self.event_type[ind]
            weather_type = self.weather_type[ind]
            event_desc = self.event_description[ind]
            weather_desc = self.weather_description[ind]
            text_parts = []
            if event_type != -1:
                text_parts.append(f"Event: {event_desc}")
            if weather_type != 0:
                text_parts.append(f"Weather: {weather_desc}")
            text = ". ".join(text_parts) if text_parts else "Normal traffic conditions."
        
        return text, x, nodelist
    
    def __len__(self):
        return self._length
    
    def __del__(self):
        if hasattr(self, 'h5_file') and self.h5_file is not None:
            self.h5_file.close()