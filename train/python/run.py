import os
import train_wav2vec2
import train_kenlm
import publish

from pathlib import Path

"""

Execute all steps for training both an acoustic and language model

"""

if __name__ == "__main__":

    perform_training_wav2vec2 = False
    perform_training_kenlm = False
    perform_optimize_kenlm = True

    #organisation = "pt001"
    models_root_dir = "/root"
    wav2vec2_model_name = "wav2vec2-xlsr-s1-portuguese"
    #language="cy"
    kenlm_model_name = "kenlm"

    wav2vec2_model_dir = os.path.join(models_root_dir, wav2vec2_model_name)
    lm_model_dir = os.path.join(models_root_dir, kenlm_model_name)

    print ("\nTraining acoustic model...")
    if perform_training_wav2vec2: wav2vec2_model_dir = train_wav2vec2.train(wav2vec2_model_dir)

    print ("\n\nTraining KenLM language model...")    
    if perform_training_kenlm: lm_model_dir = train_kenlm.train(lm_model_dir, "pt_sample_dataset.py")
    
    print ("\n\nOptimizing KenLM language model...")
    print (lm_model_dir)
    if perform_optimize_kenlm: train_kenlm.optimize(lm_model_dir, wav2vec2_model_dir, "pt_sample_dataset.py")

    print ("Packaging for publishing...")
    publish_dir = os.path.join(models_root_dir, "published", wav2vec2_model_name)

    if perform_optimize_kenlm: kenlm_archive_file_path = publish.make_model_tarfile(kenlm_model_name, lm_model_dir, publish_dir)    

    #wav2vec2_published_file_path = publish.copy_for_evaluation_or_publishing(wav2vec2_model_dir, publish_dir)
    
    #print ("Files for publication ready at {}".format(wav2vec2_published_file_path))
