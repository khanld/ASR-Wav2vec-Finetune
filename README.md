# HOW TO PRETRAIN WAV2VEC ON YOUR DATASETS
## Table of contents
1. [Documentation](#documentation)
2. [Installation](#installation)
3. [Usage](#usage)
4. [Logs and Visualization](#logs)

<a name = "documentation" ></a>
## Documentation
Most of our codes are inspired by [run_wav2vec2_pretraining_no_trainer.py](https://github.com/huggingface/transformers/blob/main/examples/pytorch/speech-pretraining/run_wav2vec2_pretraining_no_trainer.py) from Huggingface but are more friendly and easier to use, especially if you are new to Pytorch. </br>
You now can pre-train Wav2vec 2.0 model on your dataset, push it into the Huggingface hub, and finetune it on downstream tasks with just a few lines of code. Follow the below instruction on how to use it.

<a name = "installation" ></a>
## Installation
```
pip install -r requirements.txt
```

<a name = "usage" ></a>
## Usage
1. Prepare your dataset
    - Your dataset can be in <b>.txt</b> or <b>.csv</b> format.
    - <b>Path</b> column is compulsory. It contains the paths to your stored audio files. Depending on your dataset location, it can be either absolute paths or relative paths.
    - Check out our [data_example.csv](dataset/data_example.csv) file for more information.
2. Configure the config.json file
3. Run
    - Start training:
        ```
        python train.py -c config.json
        ```
    - Continue to train from resume:
        ```
        python train.py -c config.json -r
        ```
    - Load specific model and start training:
        ```
        python train.py -c config.json -p path/to/your/model.tar
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