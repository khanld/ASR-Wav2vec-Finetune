import librosa
import torch
import os
import argparse
import pandas as pd
import jiwer
import toml
from pyctcdecode import build_ctcdecoder

from transformers import Wav2Vec2CTCTokenizer, Wav2Vec2Processor, Wav2Vec2ForCTC, Wav2Vec2FeatureExtractor, Wav2Vec2ProcessorWithLM
from tqdm import tqdm


class Inferencer:
    def __init__(self, device, huggingface_folder, model_path, beam_size) -> None:
        self.device = device
        tokenizer = Wav2Vec2CTCTokenizer.from_pretrained(huggingface_folder)
        feature_extractor = Wav2Vec2FeatureExtractor.from_pretrained(huggingface_folder)
        vocabs = sorted_list = [k for k, v in sorted(tokenizer.vocab.items(), key=lambda item: item[1])]

        # print(vocabs)
        decoder = build_ctcdecoder(labels=vocabs)

        self.processor = Wav2Vec2Processor.from_pretrained(huggingface_folder)
        self.beam_processor = Wav2Vec2ProcessorWithLM(feature_extractor, tokenizer, decoder)
        self.model = Wav2Vec2ForCTC.from_pretrained(huggingface_folder).to(self.device).eval()
        self.beam_size = beam_size

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
        if isinstance(self.model, torch.nn.parallel.DistributedDataParallel):
            self.model.module.load_state_dict(checkpoint["model"], strict=True)
        else:
            self.model.load_state_dict(checkpoint["model"], strict=True)        
        print(f"Model preloaded successfully from {model_path}.")

    def beam_search(self, wav):
        input_values = self.processor(wav, sampling_rate=16000, return_tensors="pt").input_values
        logits = self.model(input_values.to(self.device)).logits[0].cpu().numpy()
        output = self.beam_processor.decode(
            logits,
            beam_width=self.beam_size

        )
        return output.text
    
    @torch.no_grad()
    def run(self, test_filepath):
        df = pd.read_csv(test_filepath, sep="|")
        decodes = []
        for path in tqdm(df['path']):
            wav, _ = librosa.load(path, sr=16000)
            decode = self.beam_search(wav)
            decodes.append(decode.lower())
        print("wer: ", jiwer.wer(df['transcript'].to_list(), decodes))
        df["decodes"] = decodes
        df.to_csv(test_filepath, sep="\t", index=False)


if __name__ == '__main__':
    args = argparse.ArgumentParser(description='ASR INFERENCE ARGS')
    args.add_argument('-f', '--test_filepath', type=str, required = True,
                      help='The tsv file contains two columns ["path", "transcript"] to benchmark. Output from model will be saved to the same file in column ["decode"]')
    args.add_argument('-s', '--huggingface_folder', type=str, default = 'huggingface-hub',
                      help='The folder where you stored the huggingface files. Check the <local_dir> argument of [huggingface.args] in config.toml. Default value: "huggingface-hub".')
    args.add_argument('-m', '--model_path', type=str, default = None,
                      help='Path to the model (.tar file) in saved/<project_name>/checkpoints. If not provided, default uses the pytorch_model.bin in the <HUGGINGFACE_FOLDER>')
    args.add_argument('-d', '--device_id', type=int, default = 0,
                      help='The device you want to test your model on if CUDA is available. Otherwise, CPU is used. Default value: 0')
    args.add_argument('-b', '--beam_size', type=int, default = 1,
                      help='If beam_size > 1, decoding process will use Beam Search for benchmark. Otherwise, greedy decoding is used')
    args = args.parse_args()
    
    device = f"cuda:{args.device_id}" if torch.cuda.is_available() else "cpu"

    inferencer = Inferencer(
        device = device, 
        huggingface_folder = args.huggingface_folder, 
        model_path = args.model_path,
        beam_size=args.beam_size)

    inferencer.run(args.test_filepath)

