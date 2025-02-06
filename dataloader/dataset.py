import sys
sys.path.append("../")
import torch

from utils.feature import load_wav
from typing import Dict

class DefaultCollate:
    def __init__(self, processor, sr) -> None:
        self.processor = processor
        self.sr = sr
    def __call__(self, inputs) -> Dict[str, torch.tensor]:
        features, transcripts = zip(*inputs)
        features, transcripts = list(features), list(transcripts)
        batch = self.processor(features, sampling_rate=16000, padding="longest", return_tensors="pt", return_attention_mask=True)

        with self.processor.as_target_processor():
            labels_batch = self.processor(transcripts, padding="longest", return_tensors="pt")

        batch["labels"] = labels_batch["input_ids"].masked_fill(labels_batch.attention_mask.ne(1), -100)
        return batch

class Dataset:
    def __init__(self, data, sr, preload_data, transform = None):
        self.data = data
        self.sr = sr
        self.transform = transform
        self.preload_data = preload_data
        
    def __len__(self) -> int:
        return len(self.data)
        
    def __getitem__(self, idx) -> tuple:
        item = self.data.iloc[idx]
        if not self.preload_data:
            feature = load_wav(item['path'], sr = self.sr)
        else:
            feature = item['wav']
        
        return feature, item['transcript']

