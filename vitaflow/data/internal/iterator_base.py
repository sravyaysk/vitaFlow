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
    def __init__(self,  hparams=None, dataset=None):
        self._hparams = HParams(hparams, self.default_hparams())
        # self.set_dataset(dataset=dataset)

    @staticmethod
    def default_hparams():
        return {
            "batch_size" : 32
        }

    @property
    def num_output_units(self):
        raise NotImplementedError

    @property
    def batch_size(self):
        raise NotImplementedError

    @property
    def num_train_samples(self):
        raise NotImplementedError


    def set_dataset(self, dataset):
        """
        Inheriting class must check for the dataset type and raise runtime error if not matched
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

    def test_input_function(self):
        """
        Returns an data set iterator function that can be used in estimator
        :return:
        """
        return self._get_test_input_function()