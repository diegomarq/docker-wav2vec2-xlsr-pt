import os
import io
import sys
import json
import yaml
import shlex
import subprocess

import torch
import torchaudio
import optuna
import text_preprocess

import librosa
import warnings

from pathlib import Path
from ctcdecode import CTCBeamDecoder
from datasets import load_dataset, load_metric, set_caching_enabled
from transformers import Wav2Vec2ForCTC, Wav2Vec2Processor

from argparse import ArgumentParser, RawTextHelpFormatter


DESCRIPTION = """

Train and optimize a KenLM language model from HuggingFace's provision of the Portuguese corpus.

"""

set_caching_enabled(False)


# Preprocessing the datasets.
def speech_file_to_array_fn(batch):
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        speech_array, sampling_rate = librosa.load(batch["audio"], sr=16_000)
    batch["speech"] = speech_array   
    batch["sampling_rate"] = sampling_rate
    return batch

def prepare_dataset(batch):
    batch["input_values"] = processor(batch["speech"], sampling_rate=batch["sampling_rate"]).input_values[0]
    batch["input_length"] = len(batch["input_values"])
    return batch


def decode(batch):
    inputs = processor(batch["speech"], sampling_rate=16_000, return_tensors="pt", padding=True)        
    with torch.no_grad():
       logits = model(inputs.input_values.to("cuda"), attention_mask=inputs.attention_mask.to("cuda")).logits

    beam_results, beam_scores, timesteps, out_lens = ctcdecoder.decode(logits)
    batch["pred_strings_with_lm"] = "".join(vocab[n] for n in beam_results[0][0][:out_lens[0][0]]).strip()

    return batch


def optimize_lm_objective(trial):    
    global ctcdecoder
    
    alpha = trial.suggest_uniform('lm_alpha', 0, 6)
    beta = trial.suggest_uniform('lm_beta',0, 5)

    try:
        binarylm_file_path=os.path.join(lm_model_dir, "lm.binary")
        ctcdecoder = CTCBeamDecoder(vocab, 
            model_path=binarylm_file_path,
            alpha=alpha,
            beta=beta,
            cutoff_top_n=40,
            cutoff_prob=1.0,
            beam_width=50,
            num_processes=3,
            blank_id=processor.tokenizer.pad_token_id,
            log_probs_input=True
        )
        result = dataset_test.map(decode)
        result_wer = wer.compute(predictions=result["pred_strings_with_lm"], references=result["sentence"])
        trial.report(result_wer, step=0)

    except Exception as e:
        print (e)
        raise

    finally:
        return result_wer 



def train(lm_dir, dataset_name):

    Path(lm_dir).mkdir(parents=True, exist_ok=True)    
    corpus_file_path = os.path.join(lm_dir, "corpus.txt")

    #corpus_name="pt_sample_dataset"

    #print ("\n Train LM from {} corpus...".format(corpus_name))
    #print ("\nLoading {} dataset...".format(dataset_name))
    #oscar_corpus = load_dataset("oscar", dataset_name)

    #print ("\nExporting PT corpus to text file {}...".format(corpus_file_path))
    #with open(corpus_file_path, 'w', encoding='utf-8') as corpus_file:
    #    for line in oscar_corpus["train"]:
    #        t = text_preprocess.cleanup(line["text"])
    #        corpus_file.write(t)

    # generate KenLM ARPA file language model
    lm_arpa_file_path=os.path.join(lm_dir, "lm_vaudimus_small.arpa.gz")
    lm_bin_file_path=os.path.join(lm_dir, "lm.binary")

    #cmd = "lmplz -o {n} --text {corpus_file} --arpa {lm_file}".format(n=5, corpus_file=corpus_file_path, lm_file=lm_arpa_file_path)
    #print (cmd)

    #subprocess.run(shlex.split(cmd), stderr=sys.stderr, stdout=sys.stdout)

    # generate binary version
    cmd = "build_binary trie {arpa_file} {bin_file}".format(arpa_file=lm_arpa_file_path, bin_file=lm_bin_file_path)
    print (cmd)

    subprocess.run(shlex.split(cmd), stderr=sys.stderr, stdout=sys.stdout)

    #
    #os.remove(corpus_file_path)
    os.remove(lm_arpa_file_path)

    return lm_dir



def optimize(lm_dir, wav2vec_model_path, dataset_name):
    global processor
    global model
    global vocab
    global wer
    global resampler
    global dataset_test
    global lm_model_dir

    lm_model_dir=lm_dir

    dataset_test = load_dataset(dataset_name, split="test")

    wer = load_metric("wer")

    processor = Wav2Vec2Processor.from_pretrained(wav2vec_model_path)
    model = Wav2Vec2ForCTC.from_pretrained(wav2vec_model_path)

    torch.cuda.empty_cache()

    model.to("cuda")

    resampler = torchaudio.transforms.Resample(48_000, 16_000)

    vocab=processor.tokenizer.convert_ids_to_tokens(range(0, processor.tokenizer.vocab_size))
    space_ix = vocab.index('|')
    vocab[space_ix]=' '

    print ("Preprocessing speech files")
    dataset_test = dataset_test.map(speech_file_to_array_fn)
    dataset_test = dataset_test.map(prepare_dataset, batch_size=8, num_proc=4)

    max_input_length_in_sec = 30.0
    dataset_test = dataset_test.filter(lambda x: x < max_input_length_in_sec * processor.feature_extractor.sampling_rate, input_columns=["input_length"])
    dataset_test = dataset_test.remove_columns(["sampling_rate", "input_values", "input_length"])

    print(f"\n NUM TEST ROWS >>{str(dataset_test.num_rows)} ")

    print ("Beginning alpha and beta hyperparameter optimization")
    study = optuna.create_study()
    study.optimize(optimize_lm_objective, n_jobs=1, n_trials=30)

    #
    lm_best = {'alpha':study.best_params['lm_alpha'], 'beta':study.best_params['lm_beta']}

    config_file_path = os.path.join(lm_model_dir, "config_ctc.yaml")
    with open (config_file_path, 'w') as config_file:
        yaml.dump(lm_best, config_file)

    print('Best params saved to config file {}: alpha={}, beta={} with WER={}'.format(config_file_path, study.best_params['lm_alpha'], study.best_params['lm_beta'], study.best_value))



def main(lm_root_dir, wav2vec2_model_path, **args):
    lm_file_path=train_kenlm(lm_root_dir, "pt_sample_dataset.py")
    optimize_kenlm(lm_file_path, wav2vec2_model_path, "pt_sample_dataset.py") 



if __name__ == "__main__":

    parser = ArgumentParser(description=DESCRIPTION, formatter_class=RawTextHelpFormatter) 

    parser.add_argument("--target_dir", dest="lm_root_dir", required=True, help="target directory for language model")
    parser.add_argument("--model", dest="wav2vec_model_path", required=True, help="acoustic model to be used for optimizing")
           
    parser.set_defaults(func=main)
    args = parser.parse_args()
    args.func(**vars(args))

