import sys
import os
import torch
import torchaudio
import json
import numpy as np
import yaml

import warnings
import librosa

import models

from argparse import ArgumentParser, RawTextHelpFormatter

from datasets import load_dataset, load_metric
from transformers import Wav2Vec2ForCTC, Wav2Vec2Processor
import text_preprocess

from shutil import copyfile
from pathlib import Path

from ctcdecode import CTCBeamDecoder

DESCRIPTION = """

Much of the code in this file was lifted from a HuggingFace blog entry:

Fine-Tune XLSR-Wav2Vec2 for low-resource ASR with Transformers
https://huggingface.co/blog/fine-tune-xlsr-wav2vec2

by Patrick von Platen

An implementation of a CTC (Connectionist Temporal Classification) beam search decoder with
KenLM language models support from https://github.com/parlance/ctcdecode has been added.
 
"""


# Preprocessing the datasets.
# We need to read the aduio files as arrays
def speech_file_to_array_fn(batch):
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        speech_array, sampling_rate = librosa.load(batch["audio"], sr=16_000)
    batch["speech"] = speech_array
    batch["time"] = float(batch["time"])
    #batch["sampling_rate"] = sampling_rate
    return batch

def prepare_dataset(batch):
    batch["input_values"] = processor(batch["speech"], sampling_rate=batch["sampling_rate"]).input_values[0]
    batch["input_length"] = len(batch["input_values"])
    return batch

def evaluate(batch):
    file_name = os.path.basename(batch["audio"])
    print(f"\n -- Evaluating {file_name}")

    inputs = processor(batch["speech"], sampling_rate=16_000, return_tensors="pt", padding=True)
    
    with torch.no_grad():
       logits = model(inputs.input_values.to("cuda"), attention_mask=inputs.attention_mask.to("cuda")).logits

    pred_ids = torch.argmax(logits, dim=-1)

    batch["pred_strings"] = processor.batch_decode(pred_ids)[0].strip()

    beam_results, beam_scores, timesteps, out_lens = ctcdecoder.decode(logits)
    pred_with_ctc = "".join(vocab[n] for n in beam_results[0][0][:out_lens[0][0]])
    batch["pred_strings_with_ctc"]=pred_with_ctc.strip()
    
    beam_results, beam_scores, timesteps, out_lens = kenlm_ctcdecoder.decode(logits)
    pred_with_lm = "".join(vocab[n] for n in beam_results[0][0][:out_lens[0][0]])
    batch["pred_strings_with_lm"]=pred_with_lm.strip()

    return batch


def main(wav2vec2_model_path, revision, **args):
    global processor
    global model
    global vocab
    global ctcdecoder
    global kenlm_ctcdecoder
    global resampler

    processor, model, vocab, ctcdecoder, kenlm_ctcdecoder = models.create(wav2vec2_model_path, revision)

    #
    dataset_test = load_dataset("pt_sample_dataset.py", split="audiotext")

    wer = load_metric("wer")

    torch.cuda.empty_cache()

    model.to("cuda")

    resampler = torchaudio.transforms.Resample(48_000, 16_000)

    print("Preprocessing speech files")
    dataset_test = dataset_test.map(speech_file_to_array_fn)
    #dataset_test = dataset_test.map(prepare_dataset, batch_size=8, num_proc=4)
    #dataset_test = dataset_test.remove_columns(["id", "time", "input_values", "sampling_rate"])

    max_input_length_in_sec = 170.0
    dataset_test = dataset_test.filter(lambda x: x < max_input_length_in_sec, input_columns=["time"])
    #dataset_test = dataset_test.filter(lambda x: x < max_input_length_in_sec * processor.feature_extractor.sampling_rate, input_columns=["input_length"])

    #dataset_test = dataset_test.remove_columns(["input_length"])

    print(f"\n NUM TEST MANUSCRIPT ROWS >> {str(dataset_test.num_rows)}")

    print("Begining evaluate")

    result = dataset_test.map(evaluate, batch_size=8)
    #result = result.filter(lambda x: x["pred_strings"] != "" and x["sentence"] != "")

    #print(f"\n RESULTING ROWS >> {str(result.num_rows)}")

    print("WER: {:2f}".format(100 * wer.compute(predictions=result["pred_strings"], references=result["sentence"])))
    print("WER with CTC: {:2f}".format(100 * wer.compute(predictions=result["pred_strings_with_ctc"], references=result["sentence"])))
    print("WER with CTC+LM: {:2f}".format(100 * wer.compute(predictions=result["pred_strings_with_lm"], references=result["sentence"])))

    sys.exit(0)


if __name__ == "__main__":
   
    models_root_dir="/models/published"
    wav2vec2_model_name = "wav2vec2-xlsr-s1-portuguese"
    #kenlm_model_name= "kenlm"

    wav2vec_model_dir = os.path.join(models_root_dir, wav2vec2_model_name)

    #src_config_ctc = os.path.join("/models/" + kenlm_model_name, "config_ctc.yaml")
    #dst_config_ctc = os.path.join(wav2vec_model_dir, "config_ctc.yaml")

    #if Path(src_config_ctc).is_file() and not Path(dst_config_ctc).is_file():
    #    copyfile(src_config_ctc, dst_config_ctc)

    parser = ArgumentParser(description=DESCRIPTION, formatter_class=RawTextHelpFormatter)
    
    parser.add_argument("--model_path", dest="wav2vec2_model_path", default=wav2vec_model_dir)
    parser.add_argument("--revision", dest="revision", default='')
    parser.set_defaults(func=main)
    args = parser.parse_args()
    args.func(**vars(args))
