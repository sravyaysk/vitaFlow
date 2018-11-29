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
Preprocessor interface for data sets
"""

__all__ = ["IPreprocessor"]
__version__ = '0.0.1'
__author__ = 'Mageswaran Dhandapani <mageswaran1989@gmail.com>'

import os

from vitaflow.config.hyperparams import HParams


class IPreprocessor():
    """
    A pre-processing interface that every dataset class should inherit.
    This make sure that data of concern are downloaded and preprocessed alike across
    all the data sets.

    Some rules are enforced here, such that it follows following directory structure:

    .. code-block:: text

        default_experiment/
            preprocessed_data/
                train/
                val/
                test/

    """

    def __init__(self, hparams=None):
        """

        :param hparams:
        """
        self._hparams = HParams(hparams, self.default_hparams())

        self.EXPERIMENT_ROOT_DIR = os.path.join(self._hparams.experiment_root_directory,
                                                self._hparams.experiment_name)
        self.DATA_OUT_DIR = os.path.join(self.EXPERIMENT_ROOT_DIR,
                                         self._hparams.preprocessed_data_path)
        self.TRAIN_OUT_PATH = os.path.join(self.DATA_OUT_DIR,
                                           self._hparams.train_data_path)
        self.VAL_OUT_PATH = os.path.join(self.DATA_OUT_DIR,
                                         self._hparams.validation_data_path)
        self.TEST_OUT_PATH = os.path.join(self.DATA_OUT_DIR,
                                          self._hparams.test_data_path)

    @staticmethod
    def default_hparams():
        """
        .. role:: python(code)
           :language: python

        .. code-block:: python

            {
                "experiment_root_directory" : os.path.expanduser("~") + "/vitaFlow/",
                "experiment_name" : "experiment_name",
                "preprocessed_data_path" : "preprocessed_data",
                "train_data_path" : "train",
                "validation_data_path" : "val",
                "test_data_path" : "test"
            }

        Here:

        "experiment_root_directory" : str
            Root directory where the data is downloaded or copied, also
            acts as the folder for any subsequent experimentation

        "experiment_name" : str
            Name for the current experiment

        "preprocessed_data_path" : str
            Folder path under `experiment_root_directory` where the preprocessed data
            should be stored

        "train_data_path" : str
            Folder path under `experiment_root_directory` where the train data is stored

        "validation_data_path" : str
            Folder path under `experiment_root_directory` where the validation data is stored

        "test_data_path" : str
            Folder path under `experiment_root_directory` where the test data is stored


        :return: A dictionary of hyperparameters with default values
        """
        return {
            "experiment_root_directory": os.path.expanduser("~") + "/vitaFlow/",
            "experiment_name": "experiment_name",
            "preprocessed_data_path": "preprocessed_data",
            "train_data_path": "train",
            "validation_data_path": "val",
            "test_data_path": "test"
        }

    def _create_target_directories(self):
        """Inheriting class must take care of creating experiment folder and all needed
        sub directories to store the preprocessed training and test data
        :return: 
        """
        raise NotImplementedError

    def _prepare_data(self):
        """Inheriting class must implement this class.
        Which not limited to, but includes:
        - Downloading or copying the data
        - Pre processing the data
        - Storing the preprocessed data as train/val/test
        :return:
        """
        raise NotImplementedError

    def preprocess_prepare(self):
        """Pre-process and prepares the data for training
        :return:
        """
        self._create_target_directories()
        self._prepare_data()
