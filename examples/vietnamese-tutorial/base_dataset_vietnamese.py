import pandas as pd
import sys
import re
import librosa
import numpy as np
from pandarallel import pandarallel
from typing import Dict, List

# For testing 
sys.path.append('..')

from sklearn.model_selection import train_test_split
from utils.feature import load_wav
from tqdm import tqdm
from torch.utils.data import Dataset
from dataloader.dataset import Dataset as InstanceDataset
from vietnam_number import n2w



class BaseDataset(Dataset):
    def __init__(self, rank, dist, path, sr, delimiter, chars_to_ignore, min_duration = -np.inf, max_duration = np.inf, preload_data = False, transform = None, nb_workers = 4):
        self.rank = rank
        self.dist = dist
        self.sr = sr
        self.chars_to_ignore = '[^\ a-zA-Z_àáãạảăắằẳẵặâấầẩẫậèéẹẻẽêềếểễệđìíĩỉịòóõọỏôốồổỗộơớờởỡợùúũụủưứừửữựỳỵỷỹýÀÁÃẠẢĂẮẰẲẴẶÂẤẦẨẪẬÈÉẸẺẼÊỀẾỂỄỆĐÌÍĨỈỊÒÓÕỌỎÔỐỒỔỖỘƠỚỜỞỠỢÙÚŨỤỦƯỨỪỬỮỰỲỴỶỸÝ]'
        self.transform = transform
        self.preload_data = preload_data
        self.min_duration = min_duration
        self.max_duration = max_duration
        self.df = self.load_data(path, delimiter)
        pandarallel.initialize(progress_bar=True, nb_workers = nb_workers)

        if min_duration != -np.inf or max_duration != np.inf:
            if self.rank == 0 and 'duration' not in self.df.columns:
                print("\n*****Generate duration column*****")
                self.df['duration'] = self.df['path'].parallel_apply(lambda filename: librosa.get_duration(filename=filename))
                self.df.to_csv(path, index = False, sep = delimiter)
            self.dist.barrier()
            self.df = self.load_data(path, delimiter)
            if self.rank == 0:
                print("\n*****Filter out invalid audio*****")
            mask = (self.df['duration'] <= self.max_duration) & (self.df['duration'] >= self.min_duration)
            self.df = self.df[mask]
        self.df['transcript'] = self.df['transcript'].parallel_apply(self.remove_special_characters)
    
        if self.preload_data:
            if self.rank == 0:
                print(f"\n*****Preloading {len(self.df)} data*****")
            self.df['wav'] = self.df['path'].parallel_apply(lambda filepath: load_wav(filepath, sr = self.sr))

    def has_numbers(self, text) -> bool:
        return any(char.isdigit() for char in text)
        
    def remove_special_characters(self, transcript) -> str:
        transcript = re.sub(self.chars_to_ignore, '', transcript).lower()
        transcript = transcript.split(' ')
        transcript = ' '.join(n2w(text.strip()) if text.strip().isnumeric() else '' if self.has_numbers(text.strip()) else text.strip() for text in transcript)
        return transcript

    def get_vocab_dict(self) -> Dict[int, str]:
        # Read https://huggingface.co/blog/fine-tune-wav2vec2-english for more information
        all_text = " ".join(list(self.df["transcript"]))
        vocab_list = list(set(all_text))
        vocab_list.sort()
        vocab_dict = {v: k for k, v in enumerate(vocab_list)}

        vocab_dict["|"] = vocab_dict[" "]
        del vocab_dict[" "]
        vocab_dict["[unk]"] = len(vocab_dict)
        vocab_dict["[pad]"] = len(vocab_dict)
        return vocab_dict

    def preload_dataset(self, paths, sr) -> List:
        wavs = []
        print("Preloading {} data".format(self.mode))
        for path in tqdm(paths, total = len(paths)):
            wav = load_wav(path, sr)
            wavs += [wav]
        return wavs

    def load_data(self, path, delimiter) -> pd.DataFrame:
        df = pd.read_csv(path, delimiter = delimiter)
        return df

    def get_data(self) -> Dataset:
        ds = InstanceDataset(self.df, self.sr, self.preload_data, self.transform)
        return ds


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