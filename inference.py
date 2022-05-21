import librosa
import torch
import os
import argparse

from transformers import Wav2Vec2Processor, Wav2Vec2ForCTC
from tqdm import tqdm


class Inferencer:
    def __init__(self, device, huggingface_folder, model_path) -> None:
        self.device = device
        self.processor = Wav2Vec2Processor.from_pretrained(huggingface_folder)
        self.model = Wav2Vec2ForCTC.from_pretrained(huggingface_folder).to(self.device)
        if model_path is not None:
            self.preload_model(model_path)


    def preload_model(self, model_path) -> None:
        """
        Preload model parameters (in "*.tar" format) at the start of experiment.
        Args:
            model_path: The file path of the *.tar file
        """
        assert os.path.exists(model_path), f"The file {model_path} is not exist. please check path."
        checkpoint = torch.load(model_path, map_location=self.device)
        self.model.load_state_dict(checkpoint["model"], strict = True)
        print(f"Model preloaded successfully from {model_path}.")


    def transcribe(self, wav) -> str:
        input_values = self.processor(wav, sampling_rate=16000, return_tensors="pt").input_values
        logits = self.model(input_values.to(self.device)).logits
        pred_ids = torch.argmax(logits, dim=-1)
        pred_transcript = self.processor.batch_decode(pred_ids)[0]
        return pred_transcript

    def run(self, test_filepath):
        filename = test_filepath.split('/')[-1].split('.')[0]
        filetype = test_filepath.split('.')[1]
        if filetype == 'txt':
            f = open(test_filepath, 'r')
            lines = f.read().splitlines()
            f.close()

            f = open(test_filepath.replace(filename, 'transcript_'+filename), 'w+')
            for line in tqdm(lines):
                wav, _ = librosa.load(line, sr = 16000)
                transcript = self.transcribe(wav)
                f.write(line + ' ' + transcript + '\n')
            f.close()

        else:
            wav, _ = librosa.load(test_filepath, sr = 16000)
            print(f"transcript: {self.transcribe(wav)}")


if __name__ == '__main__':
    args = argparse.ArgumentParser(description='ASR INFERENCE ARGS')
    args.add_argument('-f', '--test_filepath', type=str, required = True,
                      help='It can be either the path to your audio file (.wav, .mp3) or a text file (.txt) containing a list of audio file paths.')
    args.add_argument('-s', '--huggingface_folder', type=str, default = 'huggingface-hub',
                      help='The folder where you stored the huggingface files. Check the <local_dir> argument of [huggingface.args] in config.toml. Default value: "huggingface-hub".')
    args.add_argument('-m', '--model_path', type=str, default = None,
                      help='Path to the model (.tar file) in saved/<project_name>/checkpoints. If not provided, default uses the pytorch_model.bin in the <HUGGINGFACE_FOLDER>')
    args.add_argument('-d', '--device_id', type=int, default = 0,
                      help='The device you want to test your model on if CUDA is available. Otherwise, CPU is used. Default value: 0')
    args = args.parse_args()
    
    device = f"cuda:{args.device_id}" if torch.cuda.is_available() else "cpu"

    inferencer = Inferencer(
        device = device, 
        huggingface_folder = args.huggingface_folder, 
        model_path = args.model_path)

    inferencer.run(args.test_filepath)

