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

from vitaflow.config.hyperparams import HParams


class IIteratorBase():
    def __init__(self, hparams=None, dataset=None):
        self._hparams = HParams(hparams, self.default_hparams())
        # self.set_dataset(dataset=dataset)

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
        return {
            "batch_size": 32
        }

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

    def _get_test_input_function(self):
        """
        Inheriting class must implement this
        :return: callable
        """
        raise NotImplementedError

    def _get_predict_text_input_function(self, sentence):
        """

        :type sentence: str
        :param sentence:
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
        return self._get_test_input_function()

    def test_sentence_input_fn(self, sentence):
        """

        :param sentence:
        :return:
        """
        return self._get_predict_text_input_function(sentence)

    def predict_on_test_files(self, predict_fn):
        '''
        Iterate through the files and use `predict_on_test_file`, for prediction
        :param estimator: One of the models that support this data iterator
        :param df_files_path: Files that can be opened by the pandas
        :return: Creates a folder estimator.model_dir/predictions/ and adds the predicted files
        '''
        raise NotImplementedError

    def predict_on_text(self, predict_fn):
        '''
        
        :type predict_fn: object
        :param predict_fn: 
        :return: 
        '''
        raise NotImplementedError
