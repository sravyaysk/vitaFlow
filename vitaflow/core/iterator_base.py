# Copyright 2018 The vitaFlow Authors. All Rights Reserved.
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
Data iterator base class
"""

from abc import ABC
import os
import pickle

from vitaflow.core.hyperparams import HParams
from vitaflow.core import IPreprocessor
from vitaflow.helpers.os_helper import check_n_makedirs
from vitaflow.helpers.print_helper import print_info
from vitaflow.engines import Executor


class IIteratorBase(ABC):
    def __init__(self, hparams=None, dataset=None):
        self._hparams = HParams(hparams, self.default_hparams())
        # self.set_dataset(dataset=dataset)

        #TODO experiment name is not getin gupdated with actual value by this time
        # self.temp_dir = os.path.join("/tmp/vitaflow",
        #                              self._hparams.experiment_name,
        #                              type(self).__name__)
        # check_n_makedirs(self.temp_dir)

    @staticmethod
    def default_hparams():
        """
        .. role:: python(code)
           :language: python

        .. code-block:: python

            {
                "batch_size" : 32
            }

        Here:

        "batch_size" : int
            Batch size for the current iterator

        :return:  A dictionary of hyperparameters with default values
        """

        params = IPreprocessor.default_hparams()
        params.update({
            "batch_size": 32,
            "prefetch_size": 32,
        })
        return params

    @property
    def iterator_dir(self):
        """
        Returns iterator directory `experiment_root_directory`/`experiment_name`/`iterator_name`
        :return:
        """
        path = os.path.join(self._hparams.experiment_root_directory,
                            self._hparams.experiment_name,
                            type(self).__name__)
        if not os.path.exists(path):
            os.makedirs(path)
        return path

    @property
    def num_labels(self):
        raise NotImplementedError

    @property
    def batch_size(self):
        return self._hparams.batch_size

    @property
    def num_train_samples(self):
        raise NotImplementedError

    @property
    def num_val_samples(self):
        raise NotImplementedError

    @property
    def num_test_samples(self):
        raise NotImplementedError

    def set_dataset(self, dataset):
        """
        Inheriting class must check for the data set type and raise runtime error if not matched
        :type dataset: object
        :param dataset:
        :return:
        """
        raise NotImplementedError

    def _get_train_input_fn(self):
        """
        Inheriting class must implement this
        :return: callable
        """
        raise NotImplementedError

    def _get_val_input_fn(self):
        """
        Inheriting class must implement this
        :return: callable
        """
        raise NotImplementedError

    def _get_test_input_fn(self):
        """
        Inheriting class must implement this
        :return: callable
        """
        raise NotImplementedError

    def _get_predict_single_input_function(self, data):
        """

        :type data:
        :param data:
        :return:
        """
        raise NotImplementedError

    def train_input_fn(self):
        """
        Returns an data set iterator function that can be used in estimator
        :return:
        """
        return self._get_train_input_fn()

    def val_input_fn(self):
        """
        Returns an data set iterator function that can be used in estimator
        :return:
        """
        return self._get_val_input_fn()

    def test_input_fn(self):
        """
        Returns an data set iterator function that can be used in estimator
        :return:
        """
        return self._get_test_input_fn()

    def test_sentence_input_fn(self, sentence):
        """

        :param sentence:
        :return:
        """
        return self._get_predict_single_input_function(sentence)

    def predict_on_test_files(self, executor: Executor):
        """
        Iterate through the files and use `predict_on_test_files`, for prediction
        :param executor:
        :return:
        """
        raise NotImplementedError

    def predict_on_instance(self, executor: Executor, test_file_path):
        """

        :param executor:
        :param test_file_path:
        :return:
        """
        raise NotImplementedError

    def predict_on_text(self, predict_fn):
        '''
        
        :type predict_fn: object
        :param predict_fn: 
        :return: 
        '''
        raise NotImplementedError

    def store_as_pickle(self, data, file_name):
        file_path = os.path.join(self.iterator_dir, file_name)
        print_info("Writing the pickle file {}...".format(file_path))
        with open(file_path, 'wb') as f:
            pickle.dump(data, f)
        return None

    def read_pickle(self, file_name):
        file_path = os.path.join(self.iterator_dir, file_name)
        if os.path.exists(file_path):
            print_info("Reading the pickle file {}...".format(file_path))
            with open(file_path, 'rb') as f:
                data = pickle.load(f)
            return data
        else:
            return None
