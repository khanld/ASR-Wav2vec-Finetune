import pandas as pd
import os
import re
import librosa
from tqdm import tqdm

def preprocess_fpt():
    path = '../dataset/fpt/transcriptAll.txt'
    result = [','.join(['path','transcript'])]
    mp3_path = 'dataset/fpt/mp3/'
    with open(path) as f:
        lines = f.readlines()

        for line in lines:
            path, transcript = line.split('|')[:2]

            transcript = ''.join(transcript.split(','))
            
            transcript = ' '.join(re.split(r'\s+', transcript))

            
            result.append(','.join([mp3_path + path, transcript]))


        # print(len(result))
        # print(len(lines))
        f.close()

    write_file = open('../dataset/metadata/fpt.txt', 'w')
    for i in result:
        write_file.write(i + '\n')

def preprocess_vivos():
    path = '../dataset/vivos/'
    
    mp3_path = 'dataset/vivos/'

    fod = ['test','train']
    for t in fod:
        result = [','.join(['path','transcript'])]
        pa = os.path.join(path, t)
        file_path = os.path.join(pa, 'prompts.txt')


        with open(file_path) as f:
            lines = f.readlines()

            for line in lines:
                p, transcript = line.split()[0], ' '.join(line.split()[1:])

                
                result.append(','.join([mp3_path + t + '/waves' + '/' + p.split('_')[0] + '/' + p + '.wav', transcript]))


            # print(len(result))
            # print(len(lines))
            f.close()

        write_file = open('../dataset/metadata/vivos_' + t +'.txt', 'w')
        for i in result:
            write_file.write(i + '\n')


def preprocess_vlsp():
    path = '../dataset/vlsp_set_02/'
    result = [','.join(['path','transcript'])]
    wav_path = 'dataset/vlsp_set_02/'
    fod = os.listdir(path)
    for t in fod:
        if t.split('.')[-1] != 'txt':
            continue 
        
        pa = os.path.join(path, t)
        with open(pa) as f:
            lines = f.readlines()

            for line in lines:
                transcript = line

                
                result.append(','.join([wav_path + t.split('.')[0] + '.wav', transcript]))


            # print(len(result))
            # print(len(lines))
            f.close()

    write_file = open('../dataset/metadata/vlsp.txt', 'w')
    for i in result:
        write_file.write(i + '\n')


def preprocess_common_voice():
    path = '../dataset/common_voice/vi/'
    

    wav_path = 'dataset/common_voice/vi/'
    fod = os.listdir(path)
    for t in fod:
        result = [','.join(['path','transcript'])]
        if t == 'clips' or t == 'reported.tsv':
            continue 
        
        pa = os.path.join(path, t)
        with open(pa) as f:

            lines = f.readlines()

            for line in lines[1:]:
                # print(line)
                # print(line.split('\t'))
                try:
                    p, transcript = line.split('\t')[1:3]
                    transcript = ''.join(re.split(r'[\,\?\.\!\-\;\:\"]', transcript))
                except:
                    print(line)
                    continue
                result.append(','.join([wav_path + 'clips/' + p, transcript]))


            # print(len(result))
            # print(len(lines))
            f.close()

        write_file = open('../dataset/metadata/common_voice_'+ t.split('.')[0] +'.txt', 'w')
        for i in result:
            write_file.write(i + '\n')

def merge_file():
    path = '../dataset/metadata/'
    result = [';;;;;'.join(['path','transcript\n'])]

    result_test = [';;;;;'.join(['path','transcript\n'])]

    fod = os.listdir(path)
    for t in fod:
        pa = os.path.join(path, t)

        if t == 'vivos_test.txt' or t == 'common_voice_test.txt':
            with open(pa) as f:

                lines = f.readlines()

                for line in lines[1:]:
                    # print(line)
                    # print(line.split('\t'))
                    p, transcript = line.split(',')[:2]
                    transcript = ''.join(re.split(r'[\,\?\.\!\-\;\:\"]', transcript))
                    
                    result_test.append(';;;;;'.join([p, transcript]))


                # print(len(result))
                # print(len(lines))
                f.close()
        else:
            with open(pa) as f:

                lines = f.readlines()

                for line in lines[1:]:
                    # print(line)
                    # print(line.split('\t'))
                    p, transcript = line.split(',')[:2]
                    transcript = ''.join(re.split(r'[\,\?\.\!\-\;\:\"]', transcript))
                    
                    result.append(';;;;;'.join([p, transcript]))


                # print(len(result))
                # print(len(lines))
                f.close()

    write_file = open('../dataset/metadata/train.txt', 'w')
    for i in result:
        write_file.write(i)

    write_file = open('../dataset/metadata/test.txt', 'w')
    for i in result_test:
        write_file.write(i)

def prepare_vi_450h():
    f = open('/data1/speech/khanhld/ASR-Wa2vec-Finetune/dataset/vi_450h/train/text', 'r')
    lines = f.read().splitlines()
    f.close()
    texts = [tuple(line.split(' ', 1)) for line in lines]
    texts.sort(key = lambda item: item[0])
    f = open('/data1/speech/khanhld/ASR-Wa2vec-Finetune/dataset/vi_450h/train/wav.scp', 'r')
    lines = f.read().splitlines()
    f.close()
    paths = [tuple(line.split(' ', 1)) for line in lines]
    paths.sort(key = lambda item: item[0])
    
    f = open('/data1/speech/khanhld/ASR-Wa2vec-Finetune/dataset/vi_450h/train/train.txt', 'w+')
    f.write('path' + '|' + 'transcript' + '|' + 'duration' + '\n')
    for item1, item2 in tqdm(zip(texts, paths), total = len(texts)):
        fname1, text = item1
        fname2, path = item2
        duration = librosa.get_duration(filename=path)
        text = text.replace('<unk>', '')
        text = text.replace('  ', ' ')
        text = text.strip()
        assert fname1 == fname2, f"fname1 = {fname1} - fname2 = {fname2}"
        if len(text) > 0:
            f.write(path + '|' + text + '|' + str(duration) + '\n')
    f.close()

prepare_vi_450h()

# preprocess_fpt()


# preprocess_vivos()

# preprocess_vlsp()

# preprocess_common_voice()

# merge_file()