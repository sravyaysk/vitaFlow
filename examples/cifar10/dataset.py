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
Cifar10 Dataset Class
"""

import os
import shutil

from vitaflow.core import HParams
from vitaflow.core import IPreprocessor
from vitaflow.helpers.print_helper import print_info
from vitaflow.utils.data_io import maybe_download

class Cifar10Dataset(IPreprocessor):
    """
    Cifar10 Dataset Downloader and preprocessor
    Link: https://www.cs.toronto.edu/~kriz/cifar.html
    """
    def __init__(self, hparams=None):
        IPreprocessor.__init__(self, hparams=hparams)
        self._hparams = HParams(hparams, self.default_hparams())

        self._download_path = os.path.join(
            self._hparams.experiment_root_directory,
            self._hparams.experiment_name,
            "raw_data/"
        )
        self._link = "https://www.cs.toronto.edu/~kriz/cifar-10-python.tar.gz"

        maybe_download(urls=self._link,
                       path=self._download_path,
                       filenames="cifar-10-python.tar.gz",
                       extract=True)

        self._prepare_data()


    @staticmethod
    def default_hparams():
        """
        .. role:: python(code)
           :language: python

        .. code-block:: python

            {
                "experiment_root_directory" : os.path.expanduser("~") + "/vitaFlow/" ,
                "experiment_name" : "Cifiar10Dataset",
                "preprocessed_data_path" : "preprocessed_data",
                "train_data_path" : "train",
                "validation_data_path" : "val",
                "test_data_path" : "test",
                "over_write" : False,
            }

        Here:

        "experiment_root_directory" : str
            Root directory where the data is downloaded or copied, also
            acts as the folder for any subsequent experimentation

        "experiment_name" : str
            Name of the data set

        "preprocessed_data_path" : str
            Folder path under `experiment_root_directory` where the preprocessed data
            should be stored

        "train_data_path" : str
            Folder path under `experiment_root_directory` where the train data is stored

        "validation_data_path" : str
            Folder path under `experiment_root_directory` where the validation data is stored

        "test_data_path" : str
            Folder path under `experiment_root_directory` where the test data is stored

        "over_write" : boolean
            Flag to over write the previous copy of the downloaded data

        :return: A dictionary of hyperparameters with default values
        """
        hparams = IPreprocessor.default_hparams()

        hparams.update({
            "experiment_name": "Cifiar10Dataset",
        })

        return hparams

    def _create_target_directories(self):
        """
        To setup destination folders structure if not present.
        :return:
        """
        if os.path.exists(self.DATA_OUT_DIR):
            if self._hparams.over_write:
                print_info("Deleting data folder: {}".format(self.DATA_OUT_DIR))
                shutil.rmtree(self.DATA_OUT_DIR)
                print_info("Recreating data folder: {}".format(self.DATA_OUT_DIR))
                os.makedirs(self.DATA_OUT_DIR)
            else:
                print_info("Skipping preprocessing step, since the data might already be available")
        else:
            print_info("Creating data folder: {}".format(self.DATA_OUT_DIR))
            os.makedirs(self.DATA_OUT_DIR)

    def copy(self, in_path, out_dir):
        path, file_name = os.path.split(in_path)
        if not os.path.exists(out_dir):
            os.makedirs(out_dir)
        else:
            if not os.path.exists(os.path.join(out_dir, file_name)):
                print_info("Copying the file {} to {}".format(in_path, out_dir))
                shutil.copy(src=in_path, dst=out_dir)
            else:
                print_info("Found previous copy @ {}".format(os.path.join(out_dir, file_name)))

    def _prepare_data(self):
        """Inheriting class must implement this class.
        Which not limited to, but includes:
        - Downloading or copying the data
        - Pre processing the data
        - Storing the preprocessed data as train/val/test
        :return:
        """
        for file in ["data_batch_1", "data_batch_2", "data_batch_3", "data_batch_4", "data_batch_5", "batches.meta"]:
            in_path = os.path.join(self._download_path, "cifar-10-batches-py", file)
            self.copy(in_path=in_path, out_dir=self.TRAIN_OUT_PATH)
        for file in ["test_batch", "batches.meta"]:
            in_path = os.path.join(self._download_path, "cifar-10-batches-py", file)
            self.copy(in_path=in_path, out_dir=self.TEST_OUT_PATH)





"""
TODO:
- https://cntk.ai/pythondocs/CNTK_201A_CIFAR-10_DataLoader.html

"""