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
CoNLL 2003 dataset iterator class
"""

import os
import pickle

import numpy as np
import pandas as pd
import tensorflow as tf
from overrides import overrides
from tqdm import tqdm

from vitaflow.core import HParams
from vitaflow.core import IIteratorBase
from vitaflow.core import IPreprocessor
from vitaflow.core.features import ITextFeature
from vitaflow.core.nlp.spacy_helper import naive_vocab_creater, get_char_vocab, vocab_to_tsv
from vitaflow.data.text.vocabulary import SpecialTokens
from vitaflow.helpers.os_helper import check_n_makedirs
from vitaflow.helpers.print_helper import print_info
from vitaflow.engines import Executor
import csv

class CSVSeqToSeqIterator(IIteratorBase, ITextFeature):
    def __init__(self, hparams=None, dataset=None):
        '''
        :param hparams:
        :param dataset:
        '''
        IIteratorBase.__init__(self, hparams=hparams, dataset=dataset)
        ITextFeature.__init__(self)
        self._hparams = HParams(hparams, self.default_hparams())

        #
        # def _init(self):

        self.EXPERIMENT_ROOT_DIR = os.path.join(self._hparams.experiment_root_directory,
                                                self._hparams.experiment_name)
        self.PREPROCESSED_DATA_OUT_DIR = os.path.join(self.EXPERIMENT_ROOT_DIR,
                                                      self._hparams.preprocessed_data_path)
        self.TRAIN_OUT_PATH = os.path.join(self.PREPROCESSED_DATA_OUT_DIR,
                                           self._hparams.train_data_path)
        self.VAL_OUT_PATH = os.path.join(self.PREPROCESSED_DATA_OUT_DIR,
                                         self._hparams.validation_data_path)
        self.TEST_OUT_PATH = os.path.join(self.PREPROCESSED_DATA_OUT_DIR,
                                          self._hparams.test_data_path)
        self.OUT_DIR = os.path.join(self.EXPERIMENT_ROOT_DIR,
                                    self._hparams.iterator_name)

        # This rule is assumed to be correct if the previous stage is of IPreprocessor
        self.TRAIN_FILES_IN_PATH = os.path.join(self.PREPROCESSED_DATA_OUT_DIR, "train/")
        self.VAL_FILES_IN_PATH = os.path.join(self.PREPROCESSED_DATA_OUT_DIR, "val/")
        self.TEST_FILES_IN_PATH = os.path.join(self.PREPROCESSED_DATA_OUT_DIR, "test/")

        self.WORDS_VOCAB_FILE = os.path.join(self.OUT_DIR, self._hparams.text_col + "_" + "vocab.tsv")
        self.CHARS_VOCAB_FILE = os.path.join(self.OUT_DIR, self._hparams.text_col + "_" + "chars_vocab.tsv")
        self.ENTITY_VOCAB_FILE = os.path.join(self.OUT_DIR, self._hparams.entity_col + "_vocab.tsv")

        check_n_makedirs(self.OUT_DIR)

        self.padded_shapes = (tf.TensorShape([None]),  # sentence of unknown size
                              tf.TensorShape([None]),
                              tf.TensorShape([None]))  # labels of unknown size

        self.padding_values = (SpecialTokens.PAD_WORD,
                               SpecialTokens.PAD_CHAR,  # sentence padded on the right with id_pad_word
                               SpecialTokens.PAD_TAG)  # labels padded on the right with id_pad_tag

        self._extract_vocab()

    @staticmethod
    def default_hparams():
        """
        .. role:: python(code)
           :language: python

        .. code-block:: python

            {
                "experiment_root_directory" : os.path.expanduser("~") + "/vitaFlow/",
                "experiment_name" : "test_experiment",
                "iterator_name" : "conll_data_iterator",
                "preprocessed_data_path" : "preprocessed_data",
                "train_data_path" : "train",
                "validation_data_path" : "val",
                "test_data_path" : "test",
                "batch_size" : 32,
                "text_col" : 0,
                "entity_col" : 3,
            }

        Here:

        "experiment_root_directory" : str
            Root directory where the data is downloaded or copied, also
            acts as the folder for any subsequent experimentation

        "experiment_name" : str
            Name for the current experiment

        "iterator_name" : str
            Name of the data iterator

        "preprocessed_data_path" : str
            Folder path under `experiment_root_directory` where the preprocessed data
            should be stored

        "train_data_path" : str
            Folder path under `experiment_root_directory` where the train data is stored

        "validation_data_path" : str
            Folder path under `experiment_root_directory` where the validation data is stored

        "test_data_path" : str
            Folder path under `experiment_root_directory` where the test data is stored

        "batch_size" : int
            Batch size for the current iterator

        "text_col" : str
            Text column to be referred in the preprocessed CoNLL data CSV files

        "entity_col" : str
            Entity/Label column to be referred in the preprocessed CoNLL data CSV files

        "seperator" : str
            Seprator character to be used while joining the words from CSV

        "max_word_length" : int
            Maximum word length to be considerd while padding at character level

        "use_char_embd" : boolean
            Flag to enable character index as one of the feature

        :return: A dictionary of hyperparameters with default values
        """

        hparams = IPreprocessor.default_hparams()
        hparams.update(IIteratorBase.default_hparams())
        hparams.update({
            "iterator_name": "conll_data_iterator",
            "text_col": "0",
            "entity_col": "3",
            "seperator": "~",
            "quotechar": "^",
            "max_word_length": 20,
            "use_char_embd": False
        })

        return hparams

    @property
    def word_vocab_size(self):
        """
        Number of words in current data set
        """
        return self.WORD_VOCAB_SIZE

    @property
    def char_vocab_size(self):
        """
        Number of characters in the current data set
        """
        return self.CHAR_VOCAB_SIZE

    @property
    def num_lables(self):
        """
        Number of out put tags in the current data set
        """
        return self.NUM_TAGS

    @property
    def num_train_samples(self):
        """
        Number of training samples
        """
        return len(self._train_files_path)

    @property
    def num_val_samples(self):
        """
       Number of training samples
       """
        return len(self._val_files_path)

    @property
    def num_test_samples(self):
        """
       Number of training samples
       """
        return len(self._test_files_path)

    def _extract_vocab(self):
        """
        Uses the preprocessed data from the configured location and extracts
        the word and character level vocab.
        :return:
        """
        if not os.path.exists(self.WORDS_VOCAB_FILE) \
                or not os.path.exists(self.ENTITY_VOCAB_FILE) \
                or not os.path.exists(self.CHARS_VOCAB_FILE):
            print_info("Preparing the vocab for the text col: {}".format(self._hparams.text_col))

            lines = set()
            entities = set()

            for df_file in tqdm(os.listdir(self.TRAIN_FILES_IN_PATH), desc="mergining lines"):
                df_file = os.path.join(self.TRAIN_FILES_IN_PATH, df_file)

                if df_file.endswith(".csv"):
                    df = pd.read_csv(df_file,sep="\t",quoting=csv.QUOTE_NONE).fillna(SpecialTokens.UNK_WORD)
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

        self.TAGS_2_ID = {id_num: tag for id_num, tag in enumerate(tags_vocab)}

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
        for csv_file in tqdm(os.listdir(self.VAL_FILES_IN_PATH)):
            csv_file = os.path.join(self.VAL_FILES_IN_PATH, csv_file)
            files.append(csv_file)
        return files

    @property
    def _test_files_path(self):
        files = []
        for csv_file in tqdm(os.listdir(self.TEST_FILES_IN_PATH)):
            csv_file = os.path.join(self.TEST_FILES_IN_PATH, csv_file)
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
        sequence_padded, sequence_length = [], []
        if nlevels == 1:
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

            if df_file.endswith(".csv"):  # TODO start and stop tags
                df = pd.read_csv(df_file,sep="\t",quoting=csv.QUOTE_NONE).fillna(SpecialTokens.UNK_WORD)
            elif df_file.endswith(".json"):
                df = pd.read_json(df_file).filla(SpecialTokens.UNK_WORD)

            list_text = df[self._hparams.text_col].astype(str).values.tolist()
            list_char_ids = [[char_2_id_map.get(c, 0) for c in str(word)] for word in list_text]
            list_tag = df[self._hparams.entity_col].astype(str).values.tolist()
            print(list_text,list_char_ids,list_tag)
            sentence_feature1.append("{}".format(self._hparams.seperator).join(list_text))
            char_ids_feature2.append(list_char_ids)
            tag_label.append("{}".format(self._hparams.seperator).join(list_tag))

        if use_char_embd:
            sentence_feature1, seq_length = self._pad_sequences(sentence_feature1,
                                                                nlevels=1,
                                                                pad_tok="{}{}".format(self._hparams.seperator,
                                                                                      SpecialTokens.PAD_WORD))  # space is used so that it can append to the string sequence
            sentence_feature1 = np.array(sentence_feature1)

            char_ids_feature2, seq_length = self._pad_sequences(char_ids_feature2, nlevels=2,
                                                                pad_tok=int(SpecialTokens.PAD_CHAR_ID))
            char_ids_feature2 = np.array(char_ids_feature2)
            seq_length = np.array(seq_length)
            tag_label, seq_length = self._pad_sequences(tag_label,
                                                        nlevels=1,
                                                        pad_tok="{}{}".format(self._hparams.seperator,
                                                                              SpecialTokens.PAD_WORD))
            tag_label = np.array(tag_label)

            return sentence_feature1, char_ids_feature2, tag_label

        else:
            sentence_feature1 = np.array(sentence_feature1)
            tag_label = np.array(tag_label)
            return sentence_feature1, None, tag_label

    def _make_seq_pair_text(self, sentence, char_2_id_map, use_char_embd):
        '''

        :param sentence:
        :param char_2_id_map:
        :param use_char_embd:
        :return:
        '''

        # [feature1 ,feature2, label]
        sentence_feature1 = []
        char_ids_feature2 = []

        list_text = sentence.split()
        list_char_ids = [[char_2_id_map.get(c, SpecialTokens.UNK_CHAR_ID) for c in str(word)] for word in list_text]

        sentence_feature1.append("{}".format(self._hparams.seperator).join(list_text))
        char_ids_feature2.append(list_char_ids)

        if use_char_embd:
            sentence_feature1, seq_length = self._pad_sequences(sentence_feature1,
                                                                nlevels=1,
                                                                pad_tok="{}{}".format(self._hparams.seperator,
                                                                                      SpecialTokens.PAD_WORD))  # space is used so that it can append to the string sequence
            sentence_feature1 = np.array(sentence_feature1)

            char_ids_feature2, seq_length = self._pad_sequences(char_ids_feature2, nlevels=2,
                                                                pad_tok=SpecialTokens.PAD_CHAR_ID)
            char_ids_feature2 = np.array(char_ids_feature2)
            seq_length = np.array(seq_length)

            return sentence_feature1, char_ids_feature2, None

        else:
            sentence_feature1 = np.array(sentence_feature1)
            return sentence_feature1, None, None

    def get_padded_data(self, file_name):
        file_path = os.path.join(self.EXPERIMENT_ROOT_DIR , file_name)
        if os.path.exists(file_path):
            print_info("Reading the padded data...")
            with open(file_path, 'rb') as f:
                data = pickle.load(f)
            return data
        else:
            return None

    def store_padded_data(self, file_name, data):
        file_path = os.path.join(self.EXPERIMENT_ROOT_DIR , file_name)
        print_info("Writing the padded data...")
        with open(file_path, 'wb') as f:
            pickle.dump(data, f)
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

        # print_error(train_char_ids)
        # print_info(train_ner_tags)
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

        # print_error(train_char_ids)
        # print_info(train_ner_tags)
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
    def _get_test_input_fn(self):
        file_name = "test_padded_data_" + str(self._hparams.use_char_embd) + ".p"
        train_sentences, train_char_ids, train_ner_tags = None, None, None
        data = self.get_padded_data(file_name=file_name)

        if data is None:
            train_sentences, train_char_ids, train_ner_tags = \
                self._make_seq_pair(df_files_path=self.TEST_FILES_IN_PATH,
                                    char_2_id_map=self.CHAR_2_ID_MAP,
                                    use_char_embd=self._hparams.use_char_embd)
            self.store_padded_data(data=(train_sentences, train_char_ids, train_ner_tags), file_name=file_name)
        else:
            train_sentences, train_char_ids, train_ner_tags = data

        # print_error(train_char_ids)
        # print_info(train_ner_tags)
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
    def _get_predict_single_input_function(self, data):

        train_sentences, train_char_ids, train_ner_tags = None, None, None

        train_sentences, train_char_ids, train_ner_tags = \
            self._make_seq_pair_text(sentence=data,
                                     char_2_id_map=self.CHAR_2_ID_MAP,
                                     use_char_embd=self._hparams.use_char_embd)

        # print_error(train_char_ids)
        # print_info(train_ner_tags)
        if self._hparams.use_char_embd:
            dataset = tf.data.Dataset.from_tensor_slices(({self.FEATURE_1_NAME: train_sentences,
                                                           self.FEATURE_2_NAME: train_char_ids}, np.zeros(1)))
        else:
            dataset = tf.data.Dataset.from_tensor_slices(({self.FEATURE_1_NAME: train_sentences}, np.zeros(1)))
        dataset = dataset.batch(batch_size=self._hparams.batch_size)
        print_info("Dataset output sizes are: ")
        print_info(dataset.output_shapes)
        return dataset

    @overrides
    def predict_on_test_files(self, predict_fn):
        '''
        Iterate through the files and use `predict_on_test_file`, for prediction
        :param predict_fn:
        :return: Creates a folder estimator.model_dir/predictions/ and adds the predicted files
        '''

        predictions = []

        for predict in predict_fn:
            predictions.append(predict)

        results = []

        # Get the files from test folder and zip it with predictions
        for each_prediction, file in zip(predictions, self._test_files_path):

            df = pd.read_csv(file,sep="\t",quoting=csv.QUOTE_NONE)

            predicted_id = []
            confidence = []

            for tag_score in each_prediction["confidence"]:
                confidence.append(tag_score)
            for tag_id in each_prediction["viterbi_seq"]:
                predicted_id.append(self.TAGS_2_ID[tag_id])
            top_3_predicted_indices = each_prediction["top_3_indices"]
            top_3_predicted_confidence = each_prediction["top_3_confidence"]

            # print(top_3_predicted_indices)

            pred_1 = top_3_predicted_indices[:, 0:1].flatten()
            pred_1 = list(map(lambda x: self.TAGS_2_ID[x], pred_1))

            pred_2 = top_3_predicted_indices[:, 1:2].flatten()
            pred_2 = list(map(lambda x: self.TAGS_2_ID[x], pred_2))

            pred_3 = top_3_predicted_indices[:, 2:].flatten()
            pred_3 = list(map(lambda x: self.TAGS_2_ID[x], pred_3))

            pred_1_confidence = top_3_predicted_confidence[:, 0:1]
            pred_2_confidence = top_3_predicted_confidence[:, 1:2]
            pred_3_confidence = top_3_predicted_confidence[:, 2:]

            results.append({
                "tokens": df[self._hparams.text_col].astype(str).values.tolist(),
                "predicted_id": predicted_id,
                "confidence": confidence,
                "pred_1": pred_1,
                "pred_1_confidence": pred_1_confidence,
                "pred_2": pred_2,
                "pred_2_confidence": pred_2_confidence,
                "pred_3": pred_3,
                "pred_3_confidence": pred_3_confidence,

            })

            df["predicted_id"] = [j for i, j in
                                  zip(df[self._hparams.text_col].astype(str).values.tolist(), predicted_id)]
            df["confidence"]= confidence[:len(df)]
            df["pred_1"]= pred_1[:len(df)]
            df["pred_1_confidence"]= pred_1_confidence[:len(df)]
            df["pred_2"]= pred_2[:len(df)]
            df["pred_2_confidence"]= pred_2_confidence[:len(df)]

            out_dir = os.path.join(self.OUT_DIR ,"predictions")
            check_n_makedirs(out_dir)
            df.to_csv(os.path.join(out_dir ,os.path.basename(file)), index=False)

        return results

    @overrides
    def predict_on_text(self, predict_fn):
        '''

        :param predict_fn:
        :return:
        '''

        predicted_id = []
        confidence = []
        for each_prediction in predict_fn:
            for tag_score in each_prediction["confidence"]:
                confidence.append(tag_score)
            for tag_id in each_prediction["viterbi_seq"]:
                predicted_id.append(self.TAGS_2_ID[tag_id])
            top_3_predicted_indices = each_prediction["top_3_indices"]
            top_3_predicted_confidence = each_prediction["top_3_confidence"]

            # print(top_3_predicted_indices)

            pred_1 = top_3_predicted_indices[:, 0:1].flatten()
            pred_1 = list(map(lambda x: self.TAGS_2_ID[x], pred_1))

            pred_2 = top_3_predicted_indices[:, 1:2].flatten()
            pred_2 = list(map(lambda x: self.TAGS_2_ID[x], pred_2))

            pred_3 = top_3_predicted_indices[:, 2:].flatten()
            pred_3 = list(map(lambda x: self.TAGS_2_ID[x], pred_3))

            pred_1_confidence = top_3_predicted_confidence[:, 0:1]
            pred_2_confidence = top_3_predicted_confidence[:, 1:2]
            pred_3_confidence = top_3_predicted_confidence[:, 2:]

        return predicted_id


    def predict_on_test_files(self,  executor: Executor):
        """
        Runs the prediction on list of file to be tagged
        :return:
        """
        model = executor.model
        estimator = executor.estimator
        data_iterator = executor.data_iterator

        predict_fn = estimator.predict(input_fn=lambda: data_iterator.test_input_fn())

        # data_iterator.predict_on_test_files(predict_fn)

        predictions = []

        for predict in predict_fn:
            predictions.append(predict)

        results = []

        # Get the files from test folder and zip it with predictions
        for each_prediction, file in zip(predictions, self._test_files_path):

            df = pd.read_csv(file,sep="\t",quoting=csv.QUOTE_NONE)

            predicted_id = []
            confidence = []

            for tag_score in each_prediction["confidence"]:
                confidence.append(tag_score)
            for tag_id in each_prediction["viterbi_seq"]:
                predicted_id.append(self.TAGS_2_ID[tag_id])
            top_3_predicted_indices = each_prediction["top_3_indices"]
            top_3_predicted_confidence = each_prediction["top_3_confidence"]

            # print(top_3_predicted_indices)

            pred_1 = top_3_predicted_indices[:, 0:1].flatten()
            pred_1 = list(map(lambda x: self.TAGS_2_ID[x], pred_1))

            pred_2 = top_3_predicted_indices[:, 1:2].flatten()
            pred_2 = list(map(lambda x: self.TAGS_2_ID[x], pred_2))

            pred_3 = top_3_predicted_indices[:, 2:].flatten()
            pred_3 = list(map(lambda x: self.TAGS_2_ID[x], pred_3))

            pred_1_confidence = top_3_predicted_confidence[:, 0:1]
            pred_2_confidence = top_3_predicted_confidence[:, 1:2]
            pred_3_confidence = top_3_predicted_confidence[:, 2:]

            results.append({
                "tokens": df[self._hparams.text_col].astype(str).values.tolist(),
                "predicted_id": predicted_id,
                "confidence": confidence,
                "pred_1": pred_1,
                "pred_1_confidence": pred_1_confidence,
                "pred_2": pred_2,
                "pred_2_confidence": pred_2_confidence,
                "pred_3": pred_3,
                "pred_3_confidence": pred_3_confidence,

            })

            df["predicted_id"] = [j for i, j in
                                  zip(df[self._hparams.text_col].astype(str).values.tolist(), predicted_id)]

            df["confidence"]= confidence[:len(df)]
            df["pred_1"]= pred_1[:len(df)]
            df["pred_1_confidence"]= pred_1_confidence[:len(df)]
            df["pred_2"]= pred_2[:len(df)]
            df["pred_2_confidence"]= pred_2_confidence[:len(df)]

            out_dir = os.path.join(self.OUT_DIR, "predictions")
            check_n_makedirs(out_dir)
            df.to_csv(os.path.join(out_dir, os.path.basename(file)), index=False)

            prev_tag = None
            doc_text = ""
            text = ""
            enter = False
            print(file)
            for index, row in df.iterrows():
                #         print(row["0"],row["predicted_id"])
                # check if the first row
                if row["predicted_id"] != "O":
                    if index == 0 or not enter:
                        text = row["0"]
                        prev_tag = row["predicted_id"]
                        enter = True

                    else:
                        # second index onwards
                        if is_new_tag(prev_tag, row["predicted_id"]):
                            doc_text = doc_text + text + "," + strip_iob(prev_tag)+"\n"
                            text = row["0"]

                        else:
                            text = text + " " + row["0"]
                        prev_tag = row["predicted_id"]
            doc_text = doc_text + text + "," + strip_iob(prev_tag) + "\n"
            print(doc_text)

            post_out_dir = os.path.join(self.OUT_DIR, "postprocessed")
            check_n_makedirs(post_out_dir)
          
            with open(os.path.join(post_out_dir,os.path.basename(file)), "w") as post_file:
                post_file.write("Item,Tag\n")
                post_file.write(doc_text)

        # for loop ends



    #function ends        


    def predict_sentence(self,  executor: Executor, sentence):
        """
        Runs prediction on a single sentence
        :param sentence: A single sentence whose tokens are separated by space
        :return: tuple of (words_n,tags_n)
        """
        model = executor.model
        estimator = executor.estimator
        data_iterator = executor.data_iterator

        predict_fn = estimator.predict(input_fn=lambda: data_iterator.test_sentence_input_fn(sentence))
        results = data_iterator.predict_on_text(predict_fn)
        print(list(zip(sentence.split(), results)))

def strip_iob(iob_tag):
    tag = iob_tag.replace("B-", "")
    tag = tag.replace("I-", "")
    return tag

def is_new_tag(prev, current):
    if "O" in prev:
        prev_t, prev_w = "O", "O"
    else:
        prev_t, prev_w = prev.split("-")
    if "O" in current:
        current_t, current_w = "O", "O"
    else:
        current_t, current_w = current.split("-")

    if prev_w != current_w:
        return True
    else:
        if prev_t =="B" and current_t =="I":
            return False
        elif prev_t =="I" and current_t =="B":
            return True
        elif prev_t =="I" and current_t =="I":
            return False
        else:
            return False
