import librosa
import numpy as np

def load_wav(path, sr):
    return librosa.load(path, sr = sr)[0]

def subsample(data, sub_sample_length):
    assert np.ndim(data) == 1, f"Only support 1D data. The dim is {np.ndim(data)}"
    length = len(data)

    if length > sub_sample_length:
        start = np.random.randint(length - sub_sample_length)
        end = start + sub_sample_length
        data = data[start:end]
        assert len(data) == sub_sample_length
        return data
    elif length < sub_sample_length:
        data = np.append(data, np.zeros(sub_sample_length - length, dtype=np.float32))
        assert len(data) == sub_sample_length
        return data
    else:
        return data