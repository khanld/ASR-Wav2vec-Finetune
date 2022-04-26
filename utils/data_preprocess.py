import pandas as pd
import numpy as np
import os

def preprocess_vivos(mode):
    f = open(f'/content/drive/MyDrive/ASR Finetune/train.txt')
    utts = f.read().splitlines() 
    utts.sort()
    f.close()

    f = open(f'/content/drive/MyDrive/ASR Finetune/text.txt')
    prompts = f.read().splitlines() 
    prompts.sort()
    f.close()

    data = {
        "path": [],
        "transcript": []
    }
    for u, v, in zip(utts, prompts):
        u_name = u.split('/')[-1].split('.')[0].strip()
        v_name = v.split(' ', 1)[0].strip()
        txt = v.split(' ', 1)[1]
        assert u_name == v_name, f"{u_name} - {v_name}"
        data["path"] += [u]
        data["transcript"] += [txt]
    df = pd.DataFrame(data=data)
    df.to_csv(f'/content/drive/MyDrive/ASR Finetune/dataset/LibriSpeech/{mode}.csv', index=False)



preprocess_vivos('tmp')