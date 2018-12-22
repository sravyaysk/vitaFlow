"""


"""
import os
import sys
import shutil

sys.path.append(os.path.abspath('.'))

from vitaflow.core import HParams
from vitaflow.core import IPreprocessor
from vitaflow.helpers.print_helper import print_info

import tensorflow as tf


TRAIN_URL = "http://download.tensorflow.org/data/iris_training.csv"
TEST_URL = "http://download.tensorflow.org/data/iris_test.csv"

CSV_COLUMN_NAMES = ['SepalLength', 'SepalWidth',
                    'PetalLength', 'PetalWidth', 'Species']
SPECIES = ['Setosa', 'Versicolor', 'Virginica']
CSV_TYPES = [[0.0], [0.0], [0.0], [0.0], [0]]


class IrisDataset(IPreprocessor):
    def __init__(self, hparams=None):
        IPreprocessor.__init__(self, hparams=hparams)
        self._hparams = HParams(hparams, self.default_hparams())
        self._download_path = os.path.join(
            self._hparams.experiment_root_directory,
            self._hparams.experiment_name,
            "iris_data/"
        )
        from pprint import pprint
        pprint(self.__dict__)
        self._maybe_download()
        self._prepare_data()

    @staticmethod
    def default_hparams():
        # Fetching standard params
        hparams = IPreprocessor.default_hparams()
        # Additional Params
        hparams.update({
            "experiment_name": "IrisDataSet",
            "over_write": False,
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

    def _prepare_data(self):
        """Inheriting class must implement this class.
        Which not limited to, but includes:
        - Downloading or copying the data
        - Pre processing the data
        - Storing the preprocessed data as train/val/test
        :return:
        """
        print_info("Preprocessing the train data...")


    def _maybe_download(self):
        self._create_target_directories()
        train_path = tf.keras.utils.get_file(os.path.join(self._download_path, "train.txt"), TRAIN_URL)
        test_path = tf.keras.utils.get_file(os.path.join(self._download_path, "train.txt"), TEST_URL)
        val_path = tf.keras.utils.get_file(os.path.join(self._download_path, "val.txt"), TEST_URL)
        print(train_path, test_path)


t = IrisDataset()