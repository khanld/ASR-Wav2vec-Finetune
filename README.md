# FINETUNE WAV2VEC 2.0 FOR SPEECH RECOGNITION
## Table of contents
1. [Documentation](#documentation)
2. [Installation](#installation)
3. [Usage](#usage)
4. [Logs and Visualization](#logs)

<a name = "documentation" ></a>
## Documentation
Suppose you need a simple way to fine-tune the Wav2vec 2.0 model for the task of Speech Recognition on your datasets, then you came to the right place.
</br>
All documents related to this repo can be found here:
- [Wav2vec2ForCTC](https://huggingface.co/docs/transformers/model_doc/wav2vec2#transformers.Wav2Vec2ForCTC)
- [Tutorial](https://huggingface.co/blog/fine-tune-wav2vec2-english)
- [Code reference](https://github.com/huggingface/transformers/blob/main/examples/pytorch/speech-recognition/run_speech_recognition_ctc.py)


<a name = "installation" ></a>
## Installation
```
pip install -r requirements.txt
```

<a name = "usage" ></a>
## Usage
1. Prepare your dataset
    - Your dataset can be in <b>.txt</b> or <b>.csv</b> format.
    - <b>path</b> and <b>transcript</b> columns are compulsory. The <b>path</b> column contains the paths to your stored audio files, depending on your dataset location, it can be either absolute paths or relative paths. The <b>transcript</b> column contains the corresponding transcripts to the audio paths. 
    - Check out our [data_example.csv](dataset/data_example.csv) file for more information.
2. Configure the config.toml file
3. Run
    - Start training:
        ```
        python train.py -c config.toml
        ```
    - Continue to train from resume:
        ```
        python train.py -c config.toml -r
        ```
    - Load specific model and start training:
        ```
        python train.py -c config.toml -p path/to/your/model.tar
        ```

<a name = "logs" ></a>
## Logs and Visualization
The logs during the training will be stored, and you can visualize it using TensorBoard by running this command:
```
# specify the <name> in config.json
tensorboard --logdir ~/saved/<name>

# specify a port 8080
tensorboard --logdir ~/saved/<name> --port 8080
```