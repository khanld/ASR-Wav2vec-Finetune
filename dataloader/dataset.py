import sys
sys.path.append("../")

from utils.feature import load_wav

class Dataset:
    def __init__(self, data, sr, preload_data, transform = None):
        self.data = data
        self.sr = sr
        self.transform = transform
        self.preload_data = preload_data
        
    def __len__(self):
        return len(self.data)
        

    def __getitem__(self, idx):
        item = self.data.iloc[idx]
        if not self.preload_data:
            feature = load_wav(item['path'], sr = self.sr)
        else:
            feature = item['wav']
        
        return feature, item['transcript']

