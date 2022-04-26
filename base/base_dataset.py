import pandas as pd
import numpy as np
import sys
import re

# For testing 
sys.path.append('..')

from sklearn.model_selection import train_test_split
from utils.feature import load_wav
from tqdm import tqdm
from torch.utils.data import Dataset
from dataloader.dataset import Dataset as InstanceDataset

class BaseDataset(Dataset):
    def __init__(self, path, sr, preload_data, val_size = None, transform = None):
        self.df = self.load_data(path)
        self.val_size = val_size
        self.sr = sr
        self.transform = transform
        self.preload_data = preload_data
        self.df['transcript'] = self.df['transcript'].apply(self.remove_special_characters)

        if self.val_size is not None:
            assert val_size > 0 and val_size < 1, f"val_size should be greater than 0 and smaller than 1, but found {self.val_size}"
            self.train_df, self.test_df = self.split()
        else:
            self.train_df = self.df
        
    def remove_special_characters(self, transcript):
        chars_to_ignore_regex = '[\,\?\.\!\-\;\:\"]'
        return re.sub(chars_to_ignore_regex, '', transcript).lower()

    def get_vocab_dict(self):
        # Read https://huggingface.co/blog/fine-tune-wav2vec2-english for more information
        all_text = " ".join(list(self.df["transcript"]))
        vocab_list = list(set(all_text))
        vocab_list.sort()
        vocab_dict = {v: k for k, v in enumerate(vocab_list)}

        vocab_dict["|"] = vocab_dict[" "]
        del vocab_dict[" "]
        vocab_dict["[UNK]"] = len(vocab_dict)
        vocab_dict["[PAD]"] = len(vocab_dict)
        return vocab_dict

    def preload_dataset(self, paths, sr):
        wavs = []
        print("Preloading {} data".format(self.mode))
        for path in tqdm(paths, total = len(paths)):
            wav = load_wav(path, sr)
            wavs += [wav]
        return wavs

    def load_data(self, path):
        df = pd.read_csv(path)
        return df

    def split(self):
        return train_test_split(self.df, self.val_size)

    def get_data(self, mode = 'train'):
        if mode == 'train':
            if self.preload_data:
                self.train_df['wav'] = self.preload_dataset(self.train_df['path'], self.sr)
            train_ds = InstanceDataset(self.train_df, self.sr, self.preload_data, self.transform)
            return train_ds
        else:
            assert self.val_size is not None, f"val_size is not provided, cannot fetch test dataset"
            if self.preload_data:
                self.test_df['wav'] = self.preload_dataset(self.test_df['path'], self.sr)
            test_ds = InstanceDataset(self.test_df, self.sr, self.preload_data, transform = None)
            return self.test_ds


if __name__ == '__main__':
    ds = BaseDataset(
        path = '/content/drive/MyDrive/ASR Finetune/dataset/vivos/test.csv', 
        sr = 16000, 
        preload_data = False, 
        val_size = None, 
        transform = None)
    
    vocab_dict = ds.get_vocab_dict()
    for k, v in vocab_dict.items():
        print(f'{k} - {v}')