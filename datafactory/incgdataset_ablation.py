import os
import json
import numpy as np
from torch.utils.data import Dataset
from .incgdataset import INCGDataset


class INCGDatasetAblation(INCGDataset):
    """Ablation variant of INCGDataset that additionally returns event_type and weather_type"""
    
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        data_root = kwargs.get('data_root', args[1] if len(args) > 1 else './Data/MMD')
        args_obj = kwargs.get('args')
        if args_obj:
            year = args_obj.year
        else:
            year = '2017'
        
        meta_path = os.path.join(data_root, year, f'{self.period}_meta.json')
        if os.path.exists(meta_path):
            with open(meta_path, 'r') as f:
                self.meta = json.load(f)
        else:
            self.meta = None
    
    def __getitem__(self, ind):
        text, x, nodeid, timeid = super().__getitem__(ind)
        
        if self.meta is not None and ind < len(self.meta):
            event_type = self.meta[ind].get('event_type', -1)
            weather_type = self.meta[ind].get('weather_type', 0)
        else:
            example = dict(self.Data[ind])
            event_type = example.get('event_type', -1)
            weather_type = example.get('weather_type', 0)
        
        return text, x, nodeid, timeid, int(event_type), int(weather_type)
