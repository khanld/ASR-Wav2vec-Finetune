# FINETUNE WAV2VEC 2.0 FOR SPEECH RECOGNITION
[![PWC](https://img.shields.io/endpoint.svg?url=https://paperswithcode.com/badge/wav2vec2-base-vietnamese-160h/speech-recognition-on-common-voice-vi)](https://paperswithcode.com/sota/speech-recognition-on-common-voice-vi?p=wav2vec2-base-vietnamese-160h)
[![PWC](https://img.shields.io/endpoint.svg?url=https://paperswithcode.com/badge/wav2vec2-base-vietnamese-160h/speech-recognition-on-vivos)](https://paperswithcode.com/sota/speech-recognition-on-vivos?p=wav2vec2-base-vietnamese-160h)
### Table of contents
1. [Documentation](#documentation)
2. [Available Features](#feature)
3. [Installation](#installation)
4. [Usage](#usage)
5. [Logs and Visualization](#logs)
6. [Citation](#citation)
7. [Vietnamese](#vietnamese)


<a name = "documentation" ></a>
### Documentation
Suppose you need a simple way to fine-tune the Wav2vec 2.0 model for the task of Speech Recognition on your datasets, then you came to the right place.
</br>
All documents related to this repo can be found here:
- [Wav2vec2ForCTC](https://huggingface.co/docs/transformers/model_doc/wav2vec2#transformers.Wav2Vec2ForCTC)
- [Tutorial](https://huggingface.co/blog/fine-tune-wav2vec2-english)

<a name = "feature" ></a>
### Available Features
- [x] Multi-GPU training
- [x] Automatic Mix Precision
- [ ] Push to Huggingface Hub
- [ ] Continue to run at specific time step 

<a name = "installation" ></a>
### Installation
```
pip install -r requirements.txt
```

<a name = "usage" ></a>
### Usage
1. Prepare your dataset
    - Your dataset can be in <b>.txt</b> or <b>.csv</b> format.
    - <b>path</b> and <b>transcript</b> columns are compulsory. The <b>path</b> column contains the paths to your stored audio files, depending on your dataset location, it can be either absolute paths or relative paths. The <b>transcript</b> column contains the corresponding transcripts to the audio paths. 
    - Check out our [example files](examples/data_examples/) for more information.
    * <b>Important:</b> Ignoring these following notes is still OK but can hurt the performance.
        - <strong>Make sure that your transcript contains words only</strong>. Numbers should be converted into words and special characters such as ```r'[,?.!\-;:"“%\'�]'``` are removed by default,  but you can change them in the [base_dataset.py](base/base_dataset.py) if your transcript is not clean enough. 
        - If your transcript contains special tokens like ```bos_token, eos_token, unk_token (eg: <unk>, [unk],...) or pad_token (eg: <pad>, [pad],...))```. Please specify it in the [config.toml](config.toml) otherwise the Tokenizer can't recognize them.
2. Configure the [config.toml](config.toml) file
3. Run
    - Start training from scratch:
        ```
        python train.py -c config.toml
        ```
    - Resume:
        ```
        python train.py -c config.toml -r
        ```
    - Load specific model and start training:
        ```
        python train.py -c config.toml -p path/to/your/model.tar
        ```

<a name = "logs" ></a>
### Logs and Visualization
The logs during the training will be stored, and you can visualize it using TensorBoard by running this command:
```
# specify the <name> in config.json
tensorboard --logdir ~/saved/<name>

# specify a port 8080
tensorboard --logdir ~/saved/<name> --port 8080
```
![tensorboard](examples/images/tensorboard.jpeg)

<a name = "citation" ></a>
### Citation 
[![DOI](https://zenodo.org/badge/485623832.svg)](https://github.com/khanld/ASR-Wa2vec-Finetune)
```text
    @software{Duy_Khanh_Finetune_Wav2vec_2_0_2022,
    author = {Duy Khanh, Le},
    doi = {10.5281/zenodo.6540979},
    license = {CC-BY-NC-4.0},
    month = {5},
    title = {{Finetune Wav2vec 2.0 For Speech Recognition}},
    url = {https://github.com/khanld/ASR-Wa2vec-Finetune},
    year = {2022}
}
```
<a name = "vietnamese" ></a>
### Vietnamese
Please take a look [here](examples/vietnamese-tutorial) for Vietnamese people who want to train on public datasets like  [VIOS](https://huggingface.co/datasets/vivos), [COMMON VOICE](https://huggingface.co/datasets/mozilla-foundation/common_voice_8_0), [FOSD](https://data.mendeley.com/datasets/k9sxg2twv4/4), and [VLSP](https://vlsp.org.vn/vlsp2020/eval/asr).
