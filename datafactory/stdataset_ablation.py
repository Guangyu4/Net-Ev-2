import numpy as np
from .stdataset import STDataset


class STDatasetAblation(STDataset):
    """Ablation variant of STDataset that additionally returns event_type and weather_type"""
    
    def __getitem__(self, ind):
        text, x, nodelist = super().__getitem__(ind)
        event_type = int(self.event_type[ind])
        weather_type = int(self.weather_type[ind])
        return text, x, nodelist, event_type, weather_type
