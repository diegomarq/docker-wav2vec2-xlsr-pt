import os
import yaml
import tarfile
import urllib.request
from urllib.parse import urlparse

from pathlib import Path
from tqdm import tqdm

from transformers import Wav2Vec2ForCTC, Wav2Vec2Processor
from ctcdecode import CTCBeamDecoder


class DownloadProgressBar(tqdm):
    def update_to(self, b=1, bsize=1, tsize=None):
        if tsize is not None:
            self.total = tsize
        self.update(b * bsize - self.n)

       
def create(model_path, revision):

    cache_dir=model_path
    if Path(model_path).is_dir():
        processor = Wav2Vec2Processor.from_pretrained(model_path)
        model = Wav2Vec2ForCTC.from_pretrained(model_path)
    else:
        cache_dir = os.path.join('/', 'root', 'published', model_path)
        processor = Wav2Vec2Processor.from_pretrained(model_path, cache_dir=cache_dir, revision=revision)
        model = Wav2Vec2ForCTC.from_pretrained(model_path, cache_dir=cache_dir, revision=revision)
    
    targz_file_path=os.path.join(cache_dir, "kenlm.tar.gz")
    if not Path(targz_file_path).is_file():
        #print ("Downloading kenlm language model version {}".format(revision))
        #file_url = os.path.join("https://huggingface.co", model_path, "resolve", revision, 'kenlm.tar.gz')
        #download(file_url, os.path.join(cache_dir, targz_file_path))
        print ("{} not found!".format(targz_file_path))
        raise

    if not Path(os.path.join(cache_dir, "config_ctc.yaml")).is_file():
        #extract(targz_file_path)
        print ("{} not found!".format(targz_file_path))
        raise

    with open(os.path.join(cache_dir, "config_ctc.yaml"), 'r') as config_file:
        ctc_lm_params=yaml.load(config_file, Loader=yaml.FullLoader)

    vocab=processor.tokenizer.convert_ids_to_tokens(range(0, processor.tokenizer.vocab_size))
    space_ix = vocab.index('|')
    vocab[space_ix]=' '

    ctcdecoder = CTCBeamDecoder(vocab, 
        model_path='', 
        alpha=0,
        beta=0,
        cutoff_top_n=40,
        cutoff_prob=1.0,
        beam_width=50,
        num_processes=3,
        blank_id=processor.tokenizer.pad_token_id,
        log_probs_input=True
        )


    kenlm_ctcdecoder = CTCBeamDecoder(vocab,
        model_path=os.path.join(cache_dir, "lm.binary"),
        alpha=ctc_lm_params['alpha'],
        beta=ctc_lm_params['beta'],
        cutoff_top_n=40,
        cutoff_prob=1.0,
        beam_width=50,
        num_processes=4,
        blank_id=processor.tokenizer.pad_token_id,
        log_probs_input=True
    )
    
    return processor, model, vocab, ctcdecoder, kenlm_ctcdecoder


def download(file_url, output_file_path):
    with DownloadProgressBar(unit='B', unit_scale=True, miniters=1, desc=file_url.split('/')[-1]) as t:
        urllib.request.urlretrieve(file_url, filename=output_file_path, reporthook=t.update_to)

def extract(targz_file_path):
    # extract.
    if targz_file_path.endswith(".tar.gz"):
        print ("Extracting...")
        model_dir = Path(targz_file_path).parent.absolute()
        tar = tarfile.open(targz_file_path, "r:gz")
        tar.extractall(model_dir)
        tar.close()

    #Path(output_file_path).unlink()
