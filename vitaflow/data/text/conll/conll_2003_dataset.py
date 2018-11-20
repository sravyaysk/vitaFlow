# Copyright 2018 The vitFlow Authors. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#      http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""
CoNLL Dataset Class
"""

__all__ = ["CoNLL2003Dataset"]
__version__ = '0.0.1'
__author__ = 'Mageswaran Dhandapani <mageswaran1989@gmail.com>'

import os
import shutil
import pandas as pd
from tqdm import tqdm

from vitaflow.data.internal.ipreprocessor import IPreprocessor
from vitaflow.data.text.iterators.internal.dataset_types import ICoNLLType1
from vitaflow.config.hyperparams import HParams
from vitaflow.data.text.vocabulary import SpecialTokens
from vitaflow.utils.data_io import maybe_download
from vitaflow.helpers.print_helper import *

class CoNLL2003Dataset(IPreprocessor, ICoNLLType1):
    """
    CoNLL dataset obtained from https://github.com/synalp/NER/tree/master/corpus/CoNLL-2003
    and uploaded to Gdrive.

    
    """
    def __init__(self, hparams=None):
        IPreprocessor.__init__(self, hparams=hparams)
        self._hparams = HParams(hparams, self.default_hparams())
        self._download_path =  self._hparams.experiment_root_directory + \
                         "/" + self._hparams.experiment_name + \
                        "/raw_data/"
        self._link = "https://drive.google.com/open?id=1tdwxPJnnkyO-s1oHDETj89cfgLC2xp0c"

        maybe_download(urls=self._link,
                       path=self._download_path,
                       filenames="conll2003.zip",
                       extract=True)

        self.text_col = self._hparams.text_col
        self.entity_col1 = self._hparams.entity_col1
        self.entity_col2 = self._hparams.entity_col2
        self.entity_col3 = self._hparams.entity_col3

        self._prepare_data()

    @staticmethod
    def default_hparams():
        hparams = IPreprocessor.default_hparams()

        hparams.update({
            "text_col" : 0,
            "entity_col1" : 1,
            "entity_col2" : 2,
            "entity_col3" : 3,
            "number_of_word" : 5,

            "over_write" : False,
        })

        return hparams

    def _create_target_directories(self):
        if os.path.exists(self.DATA_OUT_DIR):
            if self._hparams.over_write:
                print_info("Deletingls data folder: {}".format(self.DATA_OUT_DIR))
                shutil.rmtree(self.DATA_OUT_DIR)
                print_info("Recreating data folder: {}".format(self.DATA_OUT_DIR))
                os.makedirs(self.DATA_OUT_DIR)
            else:
                print_info("Skipping preprocessing step, since the data might already be available")
        else:
            print_info("Creating data folder: {}".format(self.DATA_OUT_DIR))
            os.makedirs(self.DATA_OUT_DIR)

    def _conll_to_csv(self, txt_file_path, out_dir):
        '''
        Converts CoNLL 2003 data set text files into CSV file for each
        example/statement.
        :param txt_file_path: Input text file path
        :param out_dir: Output directory to store CSV files
        :return: Creates files in the specified train/val/test paths
        '''
        if not os.path.exists(out_dir):
            os.makedirs(out_dir)
        else:
            print_info("Found previous version at {}".format(out_dir))
            return

        print_info("Writing data to {}...".format(out_dir))

        # Read the text file
        df = pd.read_csv(txt_file_path,
                         sep=" ",
                         skip_blank_lines=False,
                         header=None).fillna(SpecialTokens.UNK_WORD)

        # Filter out the DOCSTART lines
        df = df[~df[0].str.contains("DOCSTART")]

        current_file = []

        for i in tqdm(range(len(df))):
            row = df.values[i]
            if row[0] != SpecialTokens.UNK_WORD:
                current_file.append(row)
            else:
                # Consider dumping files with size 2
                if len(current_file) > self._hparams.number_of_word:
                    current_file = pd.DataFrame(current_file)
                    current_file.to_csv(out_dir + "/{}.csv".format(i), index=False)
                    current_file = []

    def _prepare_data(self):
        print_info("Preprocessing the train data...")
        self._conll_to_csv(self._download_path + "/train.txt", self.TRAIN_OUT_PATH)

        print_info("Preprocessing the test data...")
        self._conll_to_csv(self._download_path + "/test.txt", self.TEST_OUT_PATH)

        print_info("Preprocessing the validation data...")
        self._conll_to_csv(self._download_path+"/val.txt", self.VAL_OUT_PATH)
