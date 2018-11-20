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

"""

import sys
import os
import pickle
from overrides import overrides
import numpy as np
import pandas as pd
from tqdm import tqdm
import tensorflow as tf
from overrides import overrides
import numpy as np

from vitaflow.data.text.iterators.internal.feature_types import ITextFeature
from vitaflow.data.text.vocabulary import SpecialTokens
from vitaflow.data.internal.iterator_base import IIteratorBase
from vitaflow.config.hyperparams import HParams
from vitaflow.helpers.os_helper import check_n_makedirs
from vitaflow.data.internal import IPreprocessor
from vitaflow.data.text.nlp.spacy_helper import naive_vocab_creater, get_char_vocab, vocab_to_tsv
from vitaflow.helpers.print_helper import *

class CoNLLCsvInMemory(IIteratorBase, ITextFeature):
    def __init__(self, hparams=None, dataset=None):
        '''
        Data Iterators with different features type are expected to
        implement this interface, exposing the input functions and their hooks
        :param experiment_dir:
        :param batch_size:

        '''
        IIteratorBase.__init__(self, hparams=hparams, dataset=dataset)
        ITextFeature.__init__(self)
        self._hparams = HParams(hparams, self.default_hparams())

        #
        # def _init(self):

        self.EXPERIMENT_ROOT_DIR = self._hparams.experiment_root_directory + \
                                   "/" + self._hparams.experiment_name

        self.PREPROCESSED_DATA_OUT_DIR = self.EXPERIMENT_ROOT_DIR + "/" + \
                                         self._hparams.preprocessed_data_path + "/"

        self.TRAIN_OUT_PATH = self.PREPROCESSED_DATA_OUT_DIR + "/" + \
                              self._hparams.train_data_path + "/"
        self.VAL_OUT_PATH = self.PREPROCESSED_DATA_OUT_DIR + "/" + \
                            self._hparams.validation_data_path + "/"
        self.TEST_OUT_PATH = self.PREPROCESSED_DATA_OUT_DIR + "/" + \
                             self._hparams.test_data_path + "/"

        self.OUT_DIR = self.EXPERIMENT_ROOT_DIR + "/" + self._hparams.name + "/"

        # This rule is assumed to be correct if the previous stage is of IPreprocessor
        self.TRAIN_FILES_IN_PATH = self.PREPROCESSED_DATA_OUT_DIR + "/train/"
        self.VAL_FILES_IN_PATH = self.PREPROCESSED_DATA_OUT_DIR + "/val/"
        self.TEST_FILES_IN_PATH = self.PREPROCESSED_DATA_OUT_DIR + "/test/"


        self.WORDS_VOCAB_FILE = self.OUT_DIR + "/" + self._hparams.text_col + "_" + "vocab.tsv"
        self.CHARS_VOCAB_FILE = self.OUT_DIR + "/" + self._hparams.text_col + "_" + "chars_vocab.tsv"
        self.ENTITY_VOCAB_FILE = self.OUT_DIR + "/" + self._hparams.entity_col + "_vocab.tsv"

        check_n_makedirs(self.OUT_DIR)

        self.padded_shapes = (tf.TensorShape([None]),   # sentence of unknown size
                              tf.TensorShape([None]),
                              tf.TensorShape([None]))  # labels of unknown size

        self.padding_values = (SpecialTokens.PAD_WORD,
                               SpecialTokens.PAD_CHAR,  # sentence padded on the right with id_pad_word
                               SpecialTokens.PAD_TAG)   # labels padded on the right with id_pad_tag

        self.extract_vocab()

    @staticmethod
    def default_hparams():
        """
        .. role:: python(code)
           :language: python

        .. code-block:: python

            {
            "name" : "conll_data_iterator",
            "experiment_root_directory" : "default_experiment",
            "preprocessed_data_path" : "preprocessed_data",
            "train_data_path" : "train",
            "validation_data_path" : "val",
            "test_data_path" : "test",

            "text_col" : 0,
            "entity_col" : 3,
            "batch_size" : 16,
            }

        Here:

        "name" : str
            Name of the data iterator

        "experiment_root_directory" : str
            Root directory where the data is downloaded or copied, also
            acts as the folder for any subsequent experimentation

        "preprocessed_data_path" : str
            Folder path under `experiment_root_directory` where the preprocessed data
            should be stored

        "train_data_path" : str
            Folder path under `experiment_root_directory` where the train data is stored

        "validation_data_path" : str
            Folder path under `experiment_root_directory` where the validation data is stored

        "test_data_path" : str
            Folder path under `experiment_root_directory` where the test data is stored

        "text_col" : int
            Text column to be referred in the preprocessed CoNLL data CSV files

        "entity_col" : int
            Entity/Label column to be referred in the preprocessed CoNLL data CSV files

        :return: A dictionary of hyperparameters with default values
        """

        hparams = IPreprocessor.default_hparams()

        hparams.update({
            "name" : "conll_data_iterator",
            "text_col" : "0",
            "entity_col" : "3",
            "seperator" : "~", # potential error point depending on the dataset
            "quotechar" : "^",
            "empty_line_filler" : "<LINE_END>",
            "max_word_length" : 20,
            "use_char_embd" : False
        })

        hparams.update(IIteratorBase.default_hparams())

        return hparams

    @property
    def word_vocab_size(self):
        return self.WORD_VOCAB_SIZE

    @property
    def char_vocab_size(self):
        return self.CHAR_VOCAB_SIZE

    @property
    def num_lables(self):
        return self.NUM_TAGS

    @property
    def num_train_samples(self):
        return len(self._train_files_path)

    def extract_vocab(self):
        if not os.path.exists(self.WORDS_VOCAB_FILE) \
                or not os.path.exists(self.ENTITY_VOCAB_FILE) \
                or not os.path.exists(self.CHARS_VOCAB_FILE):
            print_info("Preparing the vocab for the text col: {}".format(self._hparams.text_col))

            lines = set()
            entities = set()

            for df_file in tqdm(os.listdir(self.TRAIN_FILES_IN_PATH), desc="mergining lines"):
                df_file = os.path.join(self.TRAIN_FILES_IN_PATH, df_file)

                if df_file.endswith(".csv"):
                    df = pd.read_csv(df_file).fillna(SpecialTokens.UNK_WORD)
                else:
                    raise RuntimeError

                lines.update(set(df[self._hparams.text_col].values.tolist()))
                entities.update(set(df[self._hparams.entity_col].values.tolist()))

            self.WORD_VOCAB_SIZE, words_vocab = naive_vocab_creater(lines=lines,
                                                                    out_file_name=self.WORDS_VOCAB_FILE,
                                                                    use_nlp=True)

            print_info("Preparing the character vocab for the text col: {}".format(self._hparams.text_col))

            # Get char level vocab
            char_vocab = [SpecialTokens.PAD_CHAR, SpecialTokens.UNK_CHAR]
            _vocab = get_char_vocab(words_vocab)
            char_vocab.extend(_vocab)

            # Create char2id map
            self.CHAR_2_ID_MAP = vocab_to_tsv(vocab_list=char_vocab,
                                              out_file_name=self.CHARS_VOCAB_FILE)

            self.CHAR_VOCAB_SIZE = len(self.CHAR_2_ID_MAP)

            print_info("Preparing the vocab for the entity col: {}".format(self._hparams.entity_col))

            # NUM_TAGS, tags_vocab = tf_vocab_processor(lines, ENTITY_VOCAB_FILE)
            self.NUM_TAGS, tags_vocab = naive_vocab_creater(lines=entities,
                                                            out_file_name=self.ENTITY_VOCAB_FILE,
                                                            use_nlp=False)
        else:
            print_info("Reusing the vocab")
            self.WORD_VOCAB_SIZE, words_vocab = naive_vocab_creater(lines=None,
                                                                    out_file_name=self.WORDS_VOCAB_FILE,
                                                                    use_nlp=None)
            self.CHAR_2_ID_MAP = vocab_to_tsv(out_file_name=self.CHARS_VOCAB_FILE,
                                              vocab_list=None)
            self.CHAR_VOCAB_SIZE = len(self.CHAR_2_ID_MAP)

            self.NUM_TAGS, tags_vocab = naive_vocab_creater(lines=None,
                                                            out_file_name=self.ENTITY_VOCAB_FILE,
                                                            use_nlp=False)

        self.TAGS_2_ID =  {id_num: tag for id_num, tag in enumerate(tags_vocab)}

    @property
    def _train_files_path(self):
        files = []
        for csv_file in tqdm(os.listdir(self.TRAIN_FILES_IN_PATH)):
            csv_file = os.path.join(self.TRAIN_FILES_IN_PATH, csv_file)
            files.append(csv_file)
        return files

    @property
    def _val_files_path(self):
        files = []
        for csv_file in tqdm(os.listdir(self.TRAIN_FILES_IN_PATH)):
            csv_file = os.path.join(self.TRAIN_FILES_IN_PATH, csv_file)
            files.append(csv_file)
        return files

    def __pad_sequences(self, sequences, pad_tok, max_length):
        '''

        :param sequences:
        :param pad_tok:
        :param max_length:
        :return:
        '''
        sequence_padded, sequence_length = [], []

        for seq in sequences:
            seq = list(seq)
            seq_ = seq[:max_length] + [pad_tok] * max(max_length - len(seq), 0)
            sequence_padded += [seq_]
            sequence_length += [min(len(seq), max_length)]

        return sequence_padded, sequence_length

    def _pad_sequences(self, sequences, pad_tok, nlevels, max_word_length=10):
        '''

        :param sequences:
        :param pad_tok:
        :param nlevels:
        :param max_word_length:
        :return:
        '''
        if nlevels == 1:
            sequence_padded = []
            sequence_length = []
            max_length = max(map(lambda x: len(x.split(self._hparams.seperator)), sequences))
            # sequence_padded, sequence_length = _pad_sequences(sequences,
            #                                                   pad_tok, max_length)
            # breaking the code to pad the string instead on its ids
            for seq in sequences:
                current_length = len(seq.split(self._hparams.seperator))
                diff = max_length - current_length
                pad_data = pad_tok * diff
                sequence_padded.append(seq + pad_data)
                sequence_length.append(max_length)  # assumed

        elif nlevels == 2:
            # max_length_word = max([max(map(lambda x: len(x), seq))
            #                        for seq in sequences])
            sequence_padded, sequence_length = [], []
            for seq in tqdm(sequences):
                # all words are same length now
                sp, sl = self.__pad_sequences(seq, pad_tok, max_word_length)
                sequence_padded += [sp]
                sequence_length += [sl]

            max_length_sentence = max(map(lambda x: len(x), sequences))
            sequence_padded, _ = self.__pad_sequences(sequence_padded,
                                                      [pad_tok] * max_word_length,
                                                      max_length_sentence)
            sequence_length, _ = self.__pad_sequences(sequence_length, 0,
                                                      max_length_sentence)

        return np.array(sequence_padded), sequence_length


    def _make_seq_pair(self, df_files_path, char_2_id_map, use_char_embd):
        '''

        :param df_files_path: Path to read files that are compatible with Pandas
        :param char_2_id_map:
        :param use_char_embd:
        :return:
        '''

        # [feature1 ,feature2, label]
        sentence_feature1 = []
        char_ids_feature2 = []
        tag_label = []

        for df_file in tqdm(os.listdir(df_files_path), desc="make_seq_pair"):

            # Make the container empty
            list_text = []
            list_char_ids = []
            list_tag = []

            df_file = os.path.join(df_files_path, df_file)

            if df_file.endswith(".csv"): #TODO start and stop tags
                df = pd.read_csv(df_file).fillna(SpecialTokens.UNK_WORD)
            elif df_file.endswith(".json"):
                df = pd.read_json(df_file).filla(SpecialTokens.UNK_WORD)

            list_text = df[self._hparams.text_col].astype(str).values.tolist()
            list_char_ids = [[char_2_id_map.get(c, 0) for c in str(word)] for word in list_text]
            list_tag = df[self._hparams.entity_col].astype(str).values.tolist()

            sentence_feature1.append("{}".format(self._hparams.seperator).join(list_text))
            char_ids_feature2.append(list_char_ids)
            tag_label.append("{}".format(self._hparams.seperator).join(list_tag))


        if use_char_embd:
            sentence_feature1, seq_length = self._pad_sequences(sentence_feature1,
                                                                nlevels=1,
                                                                pad_tok="{}{}".format(self._hparams.seperator, SpecialTokens.PAD_WORD))  # space is used so that it can append to the string sequence
            sentence_feature1 = np.array(sentence_feature1)

            char_ids_feature2, seq_length = self._pad_sequences(char_ids_feature2, nlevels=2, pad_tok=int(SpecialTokens.PAD_CHAR_ID))
            char_ids_feature2 = np.array(char_ids_feature2)
            seq_length = np.array(seq_length)
            tag_label, seq_length = self._pad_sequences(tag_label,
                                                        nlevels=1,
                                                        pad_tok="{}{}".format(self._hparams.seperator, SpecialTokens.PAD_WORD))
            tag_label = np.array(tag_label)

            return sentence_feature1, char_ids_feature2, tag_label

        else:
            sentence_feature1 = np.array(sentence_feature1)
            tag_label = np.array(tag_label)
            return sentence_feature1, None, tag_label

    def get_padded_data(self, file_name):
        file_path = self.EXPERIMENT_ROOT_DIR + '/' + file_name
        if os.path.exists(file_path):
            print_info("Reading the padded data...")
            f = open(file_path, 'rb')
            data = pickle.load(f)
            f.close()
            return data
        else:
            return None

    def store_padded_data(self, file_name, data):
        file_path = self.EXPERIMENT_ROOT_DIR + '/' + file_name
        print_info("Writing the padded data...")
        f = open(file_path, 'wb')
        pickle.dump(data, f)
        f.close()
        return None

    @overrides
    def _get_train_input_fn(self):
        file_name = "train_padded_data_" + str(self._hparams.use_char_embd) + ".p"
        train_sentences, train_char_ids, train_ner_tags = None, None, None
        data = self.get_padded_data(file_name=file_name)

        if data is None:
            train_sentences, train_char_ids, train_ner_tags = \
                self._make_seq_pair(df_files_path=self.TRAIN_FILES_IN_PATH,
                                    char_2_id_map=self.CHAR_2_ID_MAP,
                                    use_char_embd=self._hparams.use_char_embd)
            self.store_padded_data(data=(train_sentences, train_char_ids, train_ner_tags), file_name=file_name)
        else:
            train_sentences, train_char_ids, train_ner_tags = data

        print_error(train_char_ids)
        print_info(train_ner_tags)
        if self._hparams.use_char_embd:
            dataset = tf.data.Dataset.from_tensor_slices(({self.FEATURE_1_NAME: train_sentences,
                                                           self.FEATURE_2_NAME: train_char_ids},
                                                          train_ner_tags))
        else:
            dataset = tf.data.Dataset.from_tensor_slices(({self.FEATURE_1_NAME: train_sentences},
                                                     train_ner_tags))
        dataset = dataset.batch(batch_size=self._hparams.batch_size)
        print_info("Dataset output sizes are: ")
        print_info(dataset.output_shapes)
        return dataset

    @overrides
    def _get_val_input_fn(self):
        file_name = "val_padded_data_" + str(self._hparams.use_char_embd) + ".p"
        train_sentences, train_char_ids, train_ner_tags = None, None, None
        data = self.get_padded_data(file_name=file_name)

        if data is None:
            train_sentences, train_char_ids, train_ner_tags = \
                self._make_seq_pair(df_files_path=self.VAL_FILES_IN_PATH,
                                    char_2_id_map=self.CHAR_2_ID_MAP,
                                    use_char_embd=self._hparams.use_char_embd)
            self.store_padded_data(data=(train_sentences, train_char_ids, train_ner_tags), file_name=file_name)
        else:
            train_sentences, train_char_ids, train_ner_tags = data

        print_error(train_char_ids)
        print_info(train_ner_tags)
        if self._hparams.use_char_embd:
            dataset = tf.data.Dataset.from_tensor_slices(({self.FEATURE_1_NAME: train_sentences,
                                                           self.FEATURE_2_NAME: train_char_ids},
                                                          train_ner_tags))
        else:
            dataset = tf.data.Dataset.from_tensor_slices(({self.FEATURE_1_NAME: train_sentences},
                                                          train_ner_tags))
        dataset = dataset.batch(batch_size=self._hparams.batch_size)
        print_info("Dataset output sizes are: ")
        print_info(dataset.output_shapes)
        return dataset

    @overrides
    def _get_test_input_function(self):
        raise NotImplementedError