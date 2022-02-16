# coding=utf-8
# Copyright 2020 The HuggingFace Datasets Authors and the current dataset script contributor.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
""" General Dataset"""


import os

import datasets

import tarfile
import zipfile
import random
from pathlib import Path

_CITATION = """\
@InProceedings{}
"""

_DESCRIPTION = """\
General datasets composed by multiple corpus with brazilian portuguese
"""

_HOMEPAGE = ""

_LICENSE = ""

_URLs = {
    'v20210920': "sample",
}


# TODO: Name of the dataset usually match the script name with CamelCase instead of snake_case
class PTBRDataset(datasets.GeneratorBasedBuilder):
    """Brazilian Portuguese Dataset"""

    VERSION = datasets.Version("0.1.0")

    # This is an example of a dataset with multiple configurations.
    # If you don't want/need to define several sub-sets in your dataset,
    # just remove the BUILDER_CONFIG_CLASS and the BUILDER_CONFIGS attributes.

    # If you need to make complex sub-parts in the datasets with configurable options
    # You can create your own builder configuration class to store attribute, inheriting from datasets.BuilderConfig
    # BUILDER_CONFIG_CLASS = MyBuilderConfig

    # You will be able to load one or the other configurations in the following list with
    # data = datasets.load_dataset('my_dataset', 'first_domain')
    BUILDER_CONFIGS = [
        datasets.BuilderConfig(
            name="v20210920", 
            version=VERSION, 
            description="""\
                Datasets: alcaim, codigodefesaconsumidor16k-master, constituicao16k-master, C-Oral-Brasil-I, cvpt, lapsbm16k-master, voxforge-ptbr, news-pt-br-82,
                date=2021-02-01,
                size=17G,
                total_hrs=162h
                """
            ),
    ]

    DEFAULT_CONFIG_NAME = "v20210920"

    def _info(self):
        
        if self.config.name == "v20210920":
            features = datasets.Features(
                {
                    "id": datasets.Value("string"),
                    "sentence": datasets.Value("string"),
                    "audio": datasets.Value("string"),
                    "time": datasets.Value("string")
                    # These are the features of your dataset like images, labels ...
                }
            )
        return datasets.DatasetInfo(
            description=_DESCRIPTION,
            features=features, 
            supervised_keys=None,
            homepage=_HOMEPAGE,
            license=_LICENSE,
            citation=_CITATION,
        )

    def _split_generators(self, dl_manager):
        """Returns SplitGenerators."""

        print (self.config.name)
        #urls = _URLs[self.config.name]
        abs_path_to_data = '/data'
        print ("\n Loading data from {}".format(abs_path_to_data))
        file_url = os.path.join(abs_path_to_data, 'midia_base.zip')

        abs_path_to_clips = os.path.join(abs_path_to_data, "wav2vec-midia-training/midia_base")
        print(abs_path_to_clips)

        #if not os.path.isdir(abs_path_to_clips):
        #    data_dir = self.download_and_extract(file_url, abs_path_to_data)

        return [
            datasets.SplitGenerator(
                name=datasets.Split.TRAIN,
                gen_kwargs={
                    "filepath": os.path.join(abs_path_to_data, "cv-valid-train-less30.csv"),
                    "path_to_clips": abs_path_to_clips,
                },
            ),
            datasets.SplitGenerator(
                name=datasets.Split.TEST,
                gen_kwargs={
                    "filepath": os.path.join(abs_path_to_data, "cv-valid-test-azure-less170.csv"),
                    "path_to_clips": abs_path_to_clips,
                },
            ),
            datasets.SplitGenerator(
                name="audiotext",
                gen_kwargs={
                    "filepath": os.path.join(abs_path_to_data, "cv-valid-test-audiotext.csv"),
                    "path_to_clips": abs_path_to_clips,
                },
            ),
            datasets.SplitGenerator(
                name="valid1",
                gen_kwargs={
                    "filepath": os.path.join(abs_path_to_data, "cv-valid-valid1-azure-less170.csv"),
                    "path_to_clips": abs_path_to_clips,
                },
            ),
            datasets.SplitGenerator(
                name="valid2",
                gen_kwargs={
                    "filepath": os.path.join(abs_path_to_data, "cv-valid-valid2-azure-less170.csv"),
                    "path_to_clips": abs_path_to_clips,
                },
            ),
            datasets.SplitGenerator(
                name="azure_audiotext",
                gen_kwargs={
                    "filepath": os.path.join(abs_path_to_data, "azure_audiotext.csv"),
                    "path_to_clips": abs_path_to_clips,
                },
            ),
        ]

    def _generate_examples(
        self, filepath, path_to_clips
    ):
        """ Yields examples. """
        data_fields = list(self._info().features.keys())
        path_idx = data_fields.index("audio")

        print("READING - " + filepath)

        CHECK_VALUES_SAMPLE = []
#        MAX_NUMBER_LINES = 3000

#        if 'sample-cv-valid-train.csv' in filepath:
#            num_lines = sum(1 for line in open(filepath))       
#            #CHECK_VALUES_SAMPLE = []
#            CHECK_VALUES_SAMPLE = random.sample(range(1, num_lines), MAX_NUMBER_LINES)
#        else:
#            CHECK_VALUES_SAMPLE = []

        with open(filepath, encoding="utf-8") as f:
            lines = f.readlines()
            headline = lines[0]

            column_names = headline.strip().split("\t")
            assert (
                column_names == data_fields
            ), f"The file should have {data_fields} as column names, but has {column_names}"

            for id_, line in enumerate(lines[1:]):

                if len(CHECK_VALUES_SAMPLE) > 0 and id_ not in CHECK_VALUES_SAMPLE:
                    continue

                field_values = line.strip().split("\t")

                if len(field_values) == 0:
                    print('WARNING - LINE EMPTY ' + str(line))
                    continue

                # set absolute path for audio file
                try:
                    audio_name = os.path.splitext(field_values[path_idx].split('\\')[-1])[0]
                    audio_file = os.path.join(path_to_clips, audio_name + '.mp3')
                    if os.path.isfile(audio_file):
                        field_values[path_idx] = audio_file

                        # if data is incomplete, fill with empty values
                        #if len(field_values) < len(data_fields):
                        #    field_values += (len(data_fields) - len(field_values)) * ["''"]

                        yield id_, {key: value for key, value in zip(data_fields, field_values)}
                    else:
                        print('WARNING - FILE NOT FOUND ' + audio_file)
                except:
                    print("ERROR - " + str(field_values) + " idx: " + str(path_idx))

    # Extra function
    def download_and_extract(self, file_url, output_file_path):
        # extract.
        if file_url.endswith(".zip"):
            print ("Extracting...")
            
            zip_ref = zipfile.ZipFile(file_url, 'r')
            zip_ref.extractall(output_file_path)
            zip_ref.close()

        elif output_file_path.endswith(".tar.gz"):
            print ("Extracting...")

            model_dir = Path(output_file_path).parent.absolute()
            tar = tarfile.open(output_file_path, "r:gz")
            tar.extractall(model_dir)
            tar.close()

        return output_file_path
