import sys
sys.path.append("../")

from utils.feature import load_wav

class DefaultCollate:
    def __init__(self, processor, sr):
        self.processor = processor
        self.sr = sr
    def __call__(self, inputs):
        features, transcripts = zip(*inputs)
        features, transcripts = list(features), list(transcripts)
        batch = self.processor(features, sampling_rate=16000, padding="longest", return_tensors="pt")

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
        
    def __len__(self):
        return len(self.data)
        

    def __getitem__(self, idx):
        item = self.data.iloc[idx]
        if not self.preload_data:
            feature = load_wav(item['path'], sr = self.sr)
        else:
            feature = item['wav']
        
        return feature, item['transcript']

